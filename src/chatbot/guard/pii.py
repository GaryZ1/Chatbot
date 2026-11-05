"""PII and secret detection / redaction (docs/13-guardrails-security.md §1-2).

Used on user input (before it reaches the LLM or the logs), at ingest (secret
scanning) and on output. The default detector is dependency-free: regular
expressions plus checksum validation (Luhn for payment cards, ISO 13616 mod-97 for
IBANs, SSA rules for US SSNs, :mod:`ipaddress` for IPs) to keep false positives
low. Redaction replaces each value with a typed placeholder (``[EMAIL_1]``); the
same value always maps to the same placeholder within one call, and the mapping
can be kept to restore the originals (reversible redaction).

:class:`PresidioDetector` is an optional adapter around Microsoft Presidio for
names, addresses and other NER-based entities (``pip install presidio-analyzer``).
"""

from __future__ import annotations

import ipaddress
import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class PIIType(StrEnum):
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    CREDIT_CARD = "CREDIT_CARD"
    IBAN = "IBAN"
    SSN = "SSN"
    IP_ADDRESS = "IP_ADDRESS"
    AWS_ACCESS_KEY = "AWS_ACCESS_KEY"
    AWS_SECRET_KEY = "AWS_SECRET_KEY"
    API_KEY = "API_KEY"
    PRIVATE_KEY = "PRIVATE_KEY"
    JWT = "JWT"
    SECRET = "SECRET"  # generic password= / token= assignments
    # NER types produced by the optional Presidio adapter
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    OTHER = "OTHER"


#: Credentials: never logged, never sent to an LLM, quarantined at ingest.
SECRET_TYPES: frozenset[PIIType] = frozenset(
    {
        PIIType.AWS_ACCESS_KEY,
        PIIType.AWS_SECRET_KEY,
        PIIType.API_KEY,
        PIIType.PRIVATE_KEY,
        PIIType.JWT,
        PIIType.SECRET,
    }
)


@dataclass(frozen=True, slots=True)
class PIIMatch:
    type: PIIType
    start: int
    end: int
    text: str
    score: float = 1.0
    detector: str = "regex"

    @property
    def is_secret(self) -> bool:
        return self.type in SECRET_TYPES


@dataclass(slots=True)
class RedactionResult:
    text: str
    matches: list[PIIMatch] = field(default_factory=list)
    #: placeholder -> original value; empty unless ``reversible=True``.
    mapping: dict[str, str] = field(default_factory=dict)

    @property
    def types(self) -> set[PIIType]:
        return {m.type for m in self.matches}

    @property
    def has_secrets(self) -> bool:
        return any(m.is_secret for m in self.matches)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for m in self.matches:
            out[m.type.value] = out.get(m.type.value, 0) + 1
        return out


# --------------------------------------------------------------------------- validators


def luhn_valid(number: str) -> bool:
    """Luhn (mod-10) checksum used by payment card numbers."""
    digits = [int(c) for c in number if c.isdigit()]
    if len(digits) < 2:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


#: ISO 13616 IBAN lengths per country (subset covering SEPA + common others).
IBAN_LENGTHS: dict[str, int] = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16, "BG": 22,
    "BH": 22, "BR": 29, "CH": 21, "CR": 22, "CY": 28, "CZ": 24, "DE": 22, "DK": 18,
    "DO": 28, "EE": 20, "EG": 29, "ES": 24, "FI": 18, "FO": 18, "FR": 27, "GB": 22,
    "GE": 22, "GI": 23, "GL": 18, "GR": 27, "GT": 28, "HR": 21, "HU": 28, "IE": 22,
    "IL": 23, "IS": 26, "IT": 27, "JO": 30, "KW": 30, "KZ": 20, "LB": 28, "LI": 21,
    "LT": 20, "LU": 20, "LV": 21, "MC": 27, "MD": 24, "ME": 22, "MK": 19, "MR": 27,
    "MT": 31, "MU": 30, "NL": 18, "NO": 15, "PK": 24, "PL": 28, "PS": 29, "PT": 25,
    "QA": 29, "RO": 24, "RS": 22, "SA": 24, "SE": 24, "SI": 19, "SK": 24, "SM": 27,
    "TN": 24, "TR": 26, "UA": 29, "VG": 24, "XK": 20,
}  # fmt: skip


def iban_valid(iban: str) -> bool:
    """ISO 13616 / ISO 7064 mod-97 check (country length checked when known)."""
    s = re.sub(r"\s+", "", iban).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return False
    expected = IBAN_LENGTHS.get(s[:2])
    if expected is not None and len(s) != expected:
        return False
    rearranged = s[4:] + s[:4]
    numeric = "".join(str(int(c, 36)) for c in rearranged)
    return int(numeric) % 97 == 1


def ssn_valid(ssn: str) -> bool:
    """US SSN structural rules (no 000/666/9xx area, no 00 group, no 0000 serial)."""
    digits = re.sub(r"\D", "", ssn)
    if len(digits) != 9:
        return False
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    if area in {"000", "666"} or area.startswith("9"):
        return False
    return group != "00" and serial != "0000"


def _ip_valid(text: str) -> bool:
    try:
        ipaddress.ip_address(text)
    except ValueError:
        return False
    return True


# --------------------------------------------------------------------------- patterns


@dataclass(frozen=True, slots=True)
class _Pattern:
    type: PIIType
    regex: re.Pattern[str]
    validator: Callable[[str], bool] | None = None
    group: int = 0  # capture group holding the sensitive value
    score: float = 1.0


def _digits_between(lo: int, hi: int) -> Callable[[str], bool]:
    def check(s: str) -> bool:
        return lo <= sum(c.isdigit() for c in s) <= hi

    return check


def _card_valid(s: str) -> bool:
    digits = re.sub(r"\D", "", s)
    return 13 <= len(digits) <= 19 and luhn_valid(digits)


# Order = priority when spans overlap (secrets first, then checksum-validated types).
_PATTERNS: tuple[_Pattern, ...] = (
    _Pattern(
        PIIType.PRIVATE_KEY,
        re.compile(
            r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----[\s\S]+?"
            r"-----END (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"
        ),
    ),
    _Pattern(PIIType.AWS_ACCESS_KEY, re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA|AIDA)[0-9A-Z]{16}\b")),
    _Pattern(
        PIIType.AWS_SECRET_KEY,
        re.compile(
            r"(?i)aws.{0,20}?(?:secret|private)?.{0,10}?key[\"']?\s*[:=]\s*[\"']?"
            r"([A-Za-z0-9/+=]{40})(?![A-Za-z0-9/+=])"
        ),
        group=1,
    ),
    _Pattern(PIIType.API_KEY, re.compile(r"\bsk-(?:ant-|proj-|live-|test-)?[A-Za-z0-9_\-]{20,}")),
    _Pattern(PIIType.API_KEY, re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b")),
    _Pattern(PIIType.API_KEY, re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b")),
    _Pattern(PIIType.API_KEY, re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    _Pattern(PIIType.API_KEY, re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    _Pattern(
        PIIType.JWT,
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    ),
    _Pattern(
        PIIType.SECRET,
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret(?:[_-]?key)?|access[_-]?token|auth[_-]?token|"
            r"token|password|passwd|pwd)\b[\"']?\s*[:=]\s*[\"']?([^\s\"',;]{8,})"
        ),
        group=1,
        score=0.8,
    ),
    _Pattern(PIIType.CREDIT_CARD, re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])"), _card_valid),
    _Pattern(
        PIIType.IBAN,
        re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}\b"),
        iban_valid,
    ),
    _Pattern(PIIType.SSN, re.compile(r"(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])"), ssn_valid),
    _Pattern(
        PIIType.SSN,
        re.compile(r"(?i)\b(?:ssn|social security(?: number)?)\s*(?:#|no\.?|:)?\s*(\d{9})\b"),
        ssn_valid,
        group=1,
    ),
    _Pattern(
        PIIType.EMAIL,
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}\b"),
    ),
    _Pattern(
        PIIType.IP_ADDRESS,
        re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"),
        _ip_valid,
    ),
    _Pattern(
        PIIType.IP_ADDRESS,
        re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}(?![\w:])|(?<![\w:])(?:[0-9A-Fa-f]{1,4}:){1,7}:(?:[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4}){0,6})?(?![\w:])|::1\b"),
        _ip_valid,
    ),
    _Pattern(
        PIIType.PHONE,
        re.compile(
            r"(?<![\w+])(?:\+\d{1,3}[\s.\-]?)?(?:\(\d{1,4}\)[\s.\-]?)?\d{2,4}(?:[\s.\-]\d{2,4}){1,4}(?![\w-])"
            r"|(?<![\w+])\+\d{8,15}\b"
        ),
    ),
)


# --------------------------------------------------------------------------- detectors


@runtime_checkable
class EntityDetector(Protocol):
    """Anything that finds sensitive spans in text."""

    def detect(self, text: str) -> list[PIIMatch]: ...


def _resolve_overlaps(matches: Iterable[tuple[int, PIIMatch]]) -> list[PIIMatch]:
    """Keep non-overlapping matches: lower priority index wins, then longer span."""
    ordered = sorted(matches, key=lambda pm: (pm[0], -(pm[1].end - pm[1].start), pm[1].start))
    taken: list[PIIMatch] = []
    for _, m in ordered:
        if any(m.start < t.end and t.start < m.end for t in taken):
            continue
        taken.append(m)
    return sorted(taken, key=lambda m: m.start)


class PIIDetector:
    """Regex + checksum PII / secret detector and redactor (the default).

    ``types`` restricts detection to a subset; ``extra`` adds detectors (e.g.
    :class:`PresidioDetector`) whose matches are merged at lower priority.
    """

    name = "regex-pii"

    def __init__(
        self,
        types: Iterable[PIIType | str] | None = None,
        *,
        min_phone_digits: int = 9,
        extra: Sequence[EntityDetector] = (),
    ) -> None:
        self.types: frozenset[PIIType] | None = (
            frozenset(PIIType(t) for t in types) if types else None
        )
        if not 7 <= min_phone_digits <= 15:
            raise ValueError("min_phone_digits must be within 7..15")
        self._phone_ok = _digits_between(min_phone_digits, 15)
        self.extra = list(extra)

    @classmethod
    def from_policy(cls, policy: Any) -> PIIDetector:
        """Build from :class:`chatbot.guard.policies.PIIPolicy`."""
        extra: list[EntityDetector] = []
        if getattr(policy, "use_presidio", False):
            try:
                extra.append(PresidioDetector())
            except ImportError:
                logger.warning("presidio requested but not installed; using regex PII only")
        return cls(
            policy.enabled_types or None, min_phone_digits=policy.min_phone_digits, extra=extra
        )

    def _enabled(self, t: PIIType) -> bool:
        return self.types is None or t in self.types

    def detect(self, text: str) -> list[PIIMatch]:
        if not text:
            return []
        found: list[tuple[int, PIIMatch]] = []
        for prio, pat in enumerate(_PATTERNS):
            if not self._enabled(pat.type):
                continue
            for m in pat.regex.finditer(text):
                value = m.group(pat.group)
                if not value:
                    continue
                if pat.validator is not None and not pat.validator(value):
                    continue
                if pat.type is PIIType.PHONE and not self._phone_ok(value):
                    continue
                start, end = m.span(pat.group)
                found.append((prio, PIIMatch(pat.type, start, end, value, pat.score)))
        base = len(_PATTERNS)
        for i, det in enumerate(self.extra):
            for em in det.detect(text):
                if self._enabled(em.type):
                    found.append((base + i, em))
        return _resolve_overlaps(found)

    def contains(self, text: str) -> bool:
        return bool(self.detect(text))

    def redact(
        self,
        text: str,
        *,
        reversible: bool = False,
        keep: Callable[[PIIMatch], bool] | None = None,
    ) -> RedactionResult:
        """Replace detected values with ``[TYPE_n]`` placeholders.

        ``keep`` lets callers whitelist matches (e.g. PII that appears in the
        sources); kept matches are not replaced and not reported.
        """
        matches = [m for m in self.detect(text) if keep is None or not keep(m)]
        if not matches:
            return RedactionResult(text=text)
        counters: dict[PIIType, int] = {}
        by_value: dict[tuple[PIIType, str], str] = {}
        mapping: dict[str, str] = {}
        parts: list[str] = []
        cursor = 0
        for m in matches:
            key = (m.type, m.text)
            placeholder = by_value.get(key)
            if placeholder is None:
                counters[m.type] = counters.get(m.type, 0) + 1
                placeholder = f"[{m.type.value}_{counters[m.type]}]"
                by_value[key] = placeholder
                if reversible:
                    mapping[placeholder] = m.text
            parts.append(text[cursor : m.start])
            parts.append(placeholder)
            cursor = m.end
        parts.append(text[cursor:])
        return RedactionResult(text="".join(parts), matches=matches, mapping=mapping)

    def redact_text(self, text: str) -> str:
        """Irreversible redaction, convenient for logs."""
        return self.redact(text).text


_PLACEHOLDER_RE = re.compile(r"\[([A-Z_]+_\d+)\]")


def restore(text: str, mapping: dict[str, str]) -> str:
    """Undo a reversible redaction (unknown placeholders are left untouched)."""
    if not mapping:
        return text
    return _PLACEHOLDER_RE.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)


class PresidioDetector:
    """Optional adapter for Microsoft Presidio (NER-based PII: names, locations ...)."""

    name = "presidio"

    _TYPE_MAP: dict[str, PIIType] = {
        "EMAIL_ADDRESS": PIIType.EMAIL,
        "PHONE_NUMBER": PIIType.PHONE,
        "CREDIT_CARD": PIIType.CREDIT_CARD,
        "IBAN_CODE": PIIType.IBAN,
        "US_SSN": PIIType.SSN,
        "IP_ADDRESS": PIIType.IP_ADDRESS,
        "PERSON": PIIType.PERSON,
        "LOCATION": PIIType.LOCATION,
    }

    def __init__(
        self,
        entities: Sequence[str] | None = ("PERSON", "LOCATION"),
        *,
        language: str = "en",
        min_score: float = 0.6,
    ) -> None:
        from presidio_analyzer import AnalyzerEngine  # optional dependency

        self._engine = AnalyzerEngine()
        self.entities = list(entities) if entities else None
        self.language = language
        self.min_score = min_score

    def detect(self, text: str) -> list[PIIMatch]:
        results = self._engine.analyze(text=text, entities=self.entities, language=self.language)
        out: list[PIIMatch] = []
        for r in results:
            if r.score < self.min_score:
                continue
            t = self._TYPE_MAP.get(r.entity_type, PIIType.OTHER)
            out.append(PIIMatch(t, r.start, r.end, text[r.start : r.end], float(r.score), self.name))
        return out


_default: PIIDetector | None = None


def default_detector() -> PIIDetector:
    """Process-wide default detector (all regex types)."""
    global _default
    if _default is None:
        _default = PIIDetector()
    return _default
