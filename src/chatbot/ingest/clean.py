"""Cleaning and normalisation (docs/01-ingestion-parsing.md §3–§4).

All functions are pure and dependency-free:

* :func:`normalize_unicode` – NFKC + removal of zero-width / control characters.
* :func:`dehyphenate` – join words hyphenated across line breaks.
* :func:`merge_broken_lines` – re-flow hard-wrapped lines while keeping Markdown
  structure (headings, lists, tables, fences, quotes).
* :func:`remove_repeated_lines` – drop page furniture (headers / footers / page
  numbers) that repeats on more than 50% of pages.
* :func:`normalize_whitespace` – collapse runs of spaces / blank lines, Markdown-safe.
* :func:`detect_language` – stopword-based language ID (+ script detection).
* :func:`clean_email` – strip quoted replies, forwarded headers and signatures.
* :func:`clean_document` – apply the above to a parsed :class:`Document`.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence

from chatbot.core.types import Block, BlockType, Document

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- unicode

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)


def normalize_unicode(text: str) -> str:
    """NFKC-normalise and drop zero-width chars, soft hyphens and C0/C1 controls (keep \\n, \\t)."""
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(
        ch for ch in text if ch in "\n\t" or unicodedata.category(ch) not in ("Cc", "Cf")
    )


# --------------------------------------------------------------------------- line repair

_HYPHEN_BREAK = re.compile(r"(\w)-[ \t]*\n[ \t]*([a-zà-ÿ])")


def dehyphenate(text: str) -> str:
    """``"infor-\\nmation"`` → ``"information"``. Only joins when the continuation is lowercase,
    so genuine compounds at line ends (``"Jean-\\nPierre"``) are preserved."""
    return _HYPHEN_BREAK.sub(r"\1\2", text)


_STRUCTURAL = re.compile(
    r"""^\s*(?:
        \#{1,6}\s            # heading
      | [-*+]\s              # bullet
      | \d{1,3}[.)]\s        # ordered item
      | \|                   # table row
      | >                    # quote
      | ```|~~~              # fence
      | !\[                  # image
      | ={3,}\s*$|-{3,}\s*$  # setext underline / rule
    )""",
    re.VERBOSE,
)
_TERMINAL = re.compile(r"[.!?:;。！？]['\")\]]*\s*$")


def _is_structural(line: str) -> bool:
    return bool(_STRUCTURAL.match(line)) or line.startswith(("    ", "\t"))


def merge_broken_lines(text: str) -> str:
    """Join hard-wrapped lines inside paragraphs; never touches Markdown structure or fences.

    A line is merged with the next one when both are non-empty, neither is
    structural, and the first does not end a sentence – or the next starts lowercase.
    """
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence or not out:
            out.append(line)
            continue
        prev = out[-1]
        can_merge = (
            prev.strip()
            and stripped
            and not _is_structural(line)
            and not prev.lstrip().startswith(("#", "|", "```", "~~~"))
            and not re.match(r"^\s*(?:={3,}|-{3,})\s*$", prev)
            and (not _TERMINAL.search(prev) or stripped[:1].islower())
        )
        # previous line being a list item is fine: wrapped continuation of the item
        if can_merge:
            out[-1] = prev.rstrip() + " " + stripped
        else:
            out.append(line)
    return "\n".join(out)


# --------------------------------------------------------------------------- page furniture

_PAGE_NUMBER = re.compile(
    r"^\s*(?:page\s*)?[-–—(\[]?\s*\d{1,4}\s*(?:(?:of|/)\s*\d{1,4})?\s*[-–—)\]]?\s*$",
    re.IGNORECASE,
)


def _furniture_key(line: str) -> str:
    return re.sub(r"\d+", "#", " ".join(line.lower().split()))


def _edge_count(n_lines: int, edge_lines: int) -> int:
    """Header/footer candidates per side; short pages only expose their first/last line."""
    return max(1, min(edge_lines, n_lines // 3))


def remove_repeated_lines(
    pages: Sequence[str], *, threshold: float = 0.5, edge_lines: int = 3, min_pages: int = 3
) -> list[str]:
    """Remove header/footer lines that repeat on more than ``threshold`` of pages.

    Only the first/last ``edge_lines`` non-empty lines of each page are candidates
    (body text that happens to repeat is kept). Digits are normalised so
    ``"ACME – Page 3"`` and ``"ACME – Page 4"`` count as the same string. Bare page
    numbers on the edges are always removed when there are at least ``min_pages``.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    if len(pages) < min_pages:
        return list(pages)
    counts: Counter[str] = Counter()
    for page in pages:
        lines = [ln for ln in page.split("\n") if ln.strip()]
        k = _edge_count(len(lines), edge_lines)
        edge = lines[:k] + lines[-k:] if lines else []
        counts.update({_furniture_key(ln) for ln in edge})
    limit = threshold * len(pages)
    repeated = {k for k, c in counts.items() if c > limit}
    out: list[str] = []
    for page in pages:
        lines = page.split("\n")
        non_empty = [i for i, ln in enumerate(lines) if ln.strip()]
        k = _edge_count(len(non_empty), edge_lines)
        edge_idx = set(non_empty[:k] + non_empty[-k:])
        kept = [
            ln
            for i, ln in enumerate(lines)
            if not (i in edge_idx and (_furniture_key(ln) in repeated or _PAGE_NUMBER.match(ln)))
        ]
        out.append("\n".join(kept).strip("\n"))
    return out


# --------------------------------------------------------------------------- whitespace

_MULTI_SPACE = re.compile(r"(?<=\S)[ \t ]{2,}")


def normalize_whitespace(text: str, *, preserve_markdown: bool = True) -> str:
    """Collapse internal runs of spaces, strip trailing spaces and limit blank lines to one.

    With ``preserve_markdown`` leading indentation (nested lists, indented code) and
    the contents of fenced code blocks are kept verbatim; pipe-table rows keep
    their cell padding collapsed but stay one row per line.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    in_fence = False
    for line in lines:
        if preserve_markdown and line.strip().startswith(("```", "~~~")):
            in_fence = not in_fence
            out.append(line.rstrip())
            continue
        if in_fence:
            out.append(line.rstrip())
            continue
        if preserve_markdown:
            indent = len(line) - len(line.lstrip(" \t"))
            body = _MULTI_SPACE.sub(" ", line[indent:].replace("\t", " ")).rstrip()
            out.append((line[:indent].replace("\t", "    ") + body) if body else "")
        else:
            out.append(" ".join(line.split()))
    joined = "\n".join(out)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip("\n")


# --------------------------------------------------------------------------- language

_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
        [
            "the",
            "of",
            "and",
            "to",
            "in",
            "is",
            "that",
            "it",
            "for",
            "on",
            "with",
            "as",
            "are",
            "was",
            "this",
            "be",
            "by",
            "or",
            "not",
            "from",
            "at",
            "which",
            "you",
            "an",
            "have",
            "can",
            "will",
            "your",
            "we",
            "they",
            "has",
            "if",
            "all",
            "but",
        ]
    ),
    "de": frozenset(
        [
            "der",
            "die",
            "das",
            "und",
            "ist",
            "nicht",
            "zu",
            "den",
            "von",
            "mit",
            "sich",
            "des",
            "auf",
            "für",
            "im",
            "dem",
            "ein",
            "eine",
            "auch",
            "es",
            "an",
            "werden",
            "aus",
            "er",
            "hat",
            "dass",
            "sie",
            "nach",
            "wird",
            "bei",
            "oder",
            "wir",
        ]
    ),
    "fr": frozenset(
        [
            "le",
            "la",
            "les",
            "de",
            "des",
            "et",
            "est",
            "un",
            "une",
            "du",
            "en",
            "que",
            "qui",
            "dans",
            "pour",
            "pas",
            "sur",
            "au",
            "avec",
            "ce",
            "il",
            "elle",
            "sont",
            "par",
            "plus",
            "ne",
            "nous",
            "vous",
            "mais",
            "ou",
            "cette",
        ]
    ),
    "es": frozenset(
        [
            "el",
            "la",
            "los",
            "las",
            "de",
            "del",
            "y",
            "en",
            "que",
            "es",
            "un",
            "una",
            "por",
            "con",
            "para",
            "no",
            "se",
            "su",
            "al",
            "lo",
            "como",
            "más",
            "pero",
            "sus",
            "le",
            "ya",
            "o",
            "este",
            "está",
            "son",
        ]
    ),
    "it": frozenset(
        [
            "il",
            "lo",
            "la",
            "gli",
            "le",
            "di",
            "e",
            "che",
            "è",
            "un",
            "una",
            "per",
            "non",
            "con",
            "del",
            "della",
            "in",
            "si",
            "da",
            "sono",
            "al",
            "come",
            "più",
            "ma",
            "anche",
            "questo",
            "nel",
            "alla",
        ]
    ),
    "nl": frozenset(
        [
            "de",
            "het",
            "een",
            "en",
            "van",
            "is",
            "dat",
            "in",
            "te",
            "op",
            "niet",
            "zijn",
            "met",
            "voor",
            "er",
            "maar",
            "ook",
            "als",
            "aan",
            "bij",
            "door",
            "om",
            "dan",
            "worden",
            "wordt",
            "deze",
        ]
    ),
    "pt": frozenset(
        [
            "o",
            "a",
            "os",
            "as",
            "de",
            "do",
            "da",
            "dos",
            "das",
            "e",
            "em",
            "que",
            "é",
            "um",
            "uma",
            "para",
            "com",
            "não",
            "se",
            "por",
            "mais",
            "ao",
            "como",
            "mas",
            "foi",
            "no",
            "na",
            "são",
        ]
    ),
}
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _script_language(text: str) -> str | None:
    counts: Counter[str] = Counter()
    for ch in text:
        o = ord(ch)
        if 0x3040 <= o <= 0x30FF:
            counts["ja"] += 1
        elif 0xAC00 <= o <= 0xD7AF:
            counts["ko"] += 1
        elif 0x4E00 <= o <= 0x9FFF:
            counts["zh"] += 1
        elif 0x0400 <= o <= 0x04FF:
            counts["ru"] += 1
        elif 0x0600 <= o <= 0x06FF:
            counts["ar"] += 1
        elif 0x0590 <= o <= 0x05FF:
            counts["he"] += 1
        elif 0x0370 <= o <= 0x03FF:
            counts["el"] += 1
        elif ch.isalpha():
            counts["latin"] += 1
    if not counts:
        return None
    if counts["ja"] and counts["ja"] + counts["zh"] > counts["latin"]:
        return "ja"  # kana present → Japanese even with many kanji
    lang, n = counts.most_common(1)[0]
    if lang == "latin" or n < 0.3 * sum(counts.values()):
        return None
    return lang


def detect_language(text: str, *, default: str = "und", min_words: int = 5) -> tuple[str, float]:
    """Return ``(iso639-1 code, confidence)``; ``default`` ("und") when undecidable.

    Non-Latin scripts are identified by Unicode block; Latin-script languages by
    the share of each language's stopwords among the words (a cheap, robust
    heuristic for per-document *and* per-section routing).
    """
    sample = text[:20000]
    script = _script_language(sample)
    if script:
        return script, 0.9
    words = [w.lower() for w in _WORD.findall(sample)]
    if len(words) < min_words:
        return default, 0.0
    scores = {lang: sum(1 for w in words if w in sw) for lang, sw in _STOPWORDS.items()}
    best, best_hits = max(scores.items(), key=lambda kv: kv[1])
    if best_hits == 0:
        return default, 0.0
    ranked = sorted(scores.values(), reverse=True)
    margin = (ranked[0] - ranked[1]) / ranked[0] if len(ranked) > 1 else 1.0
    density = best_hits / len(words)
    confidence = round(min(1.0, density * 3) * (0.5 + 0.5 * margin), 3)
    if density < 0.05:
        return default, confidence
    return best, confidence


# --------------------------------------------------------------------------- email / tickets

_REPLY_HEADER = re.compile(
    r"^\s*(?:On\s.{0,200}?\swrote:|Am\s.{0,200}?\sschrieb.{0,40}:|Le\s.{0,200}?\sa\sécrit\s?:)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_ORIGINAL_MSG = re.compile(
    r"^\s*(?:-{2,}\s*(?:Original Message|Forwarded message)\s*-{2,}|_{10,})\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_OUTLOOK_HEADER = re.compile(r"^\s*From:\s.+\n(?:\s*\w[\w -]*:.*\n){1,6}", re.MULTILINE)
_SIGNOFF = re.compile(
    r"^\s*(?:best(?: regards)?|kind regards|regards|cheers|thanks(?: again)?|thank you|"
    r"many thanks|sincerely|br|mit freundlichen grüßen|cordialement)[,.!]?\s*$",
    re.IGNORECASE,
)
_MOBILE_FOOTER = re.compile(r"^\s*Sent from my \w+.*$", re.IGNORECASE | re.MULTILINE)


def clean_email(text: str, *, max_signature_lines: int = 6) -> str:
    """Strip quoted replies, forwarded/original-message trails and signatures.

    Quoted text otherwise dominates ticket and email indexes (docs/01 §3).
    """
    text = text.replace("\r\n", "\n")
    cut = len(text)
    for pat in (_REPLY_HEADER, _ORIGINAL_MSG, _OUTLOOK_HEADER):
        m = pat.search(text)
        if m and m.start() < cut and m.start() > 0:
            cut = m.start()
    text = text[:cut]
    text = _MOBILE_FOOTER.sub("", text)
    lines = [ln for ln in text.split("\n") if not ln.lstrip().startswith(">")]
    # RFC 3676 signature delimiter
    for i, ln in enumerate(lines):
        if ln.rstrip() == "--" or ln == "-- ":
            lines = lines[:i]
            break
    # sign-off followed by a short block (name, title, phone)
    for i in range(len(lines) - 1, -1, -1):
        if _SIGNOFF.match(lines[i]):
            tail = [ln for ln in lines[i + 1 :] if ln.strip()]
            if len(tail) <= max_signature_lines and all(len(t) < 80 for t in tail):
                lines = lines[:i]
            break
    return normalize_whitespace("\n".join(lines)).strip()


# --------------------------------------------------------------------------- documents

_TEXT_BLOCKS = {BlockType.PARAGRAPH, BlockType.LIST, BlockType.HEADING, BlockType.FIGURE}


def _strip_page_furniture(blocks: list[Block], threshold: float) -> list[Block]:
    pages: dict[int, list[int]] = defaultdict(list)
    for i, b in enumerate(blocks):
        if b.page is not None and b.type in _TEXT_BLOCKS:
            pages[b.page].append(i)
    if len(pages) < 3:
        return blocks
    order = sorted(pages)
    page_texts = ["\n".join(blocks[i].text for i in pages[p]) for p in order]
    cleaned = remove_repeated_lines(page_texts, threshold=threshold)
    if cleaned == page_texts:
        return blocks
    # Re-split the cleaned page text back over the page's blocks by line count.
    new_blocks = list(blocks)
    for p, cleaned_text in zip(order, cleaned, strict=True):
        kept_lines = cleaned_text.split("\n") if cleaned_text else []
        kept_counter: Counter[str] = Counter(kept_lines)
        for i in pages[p]:
            lines = blocks[i].text.split("\n")
            keep: list[str] = []
            for ln in lines:
                if kept_counter[ln] > 0:
                    kept_counter[ln] -= 1
                    keep.append(ln)
            new_blocks[i] = blocks[i].model_copy(update={"text": "\n".join(keep).strip()})
    return new_blocks


def annotate_section_languages(doc: Document, *, min_words: int = 20) -> None:
    """Set ``block.metadata['language']`` per heading-delimited section (in place)."""
    section: list[Block] = []

    def flush() -> None:
        text = " ".join(b.text for b in section)
        if len(text.split()) >= min_words:
            lang, _ = detect_language(text)
            if lang != "und":
                for b in section:
                    b.metadata["language"] = lang
        section.clear()

    for b in doc.blocks:
        if b.type == BlockType.HEADING:
            flush()
        section.append(b)
    flush()


def clean_document(
    doc: Document,
    *,
    email: bool = False,
    furniture_threshold: float = 0.5,
    detect_lang: bool = True,
) -> Document:
    """Return a cleaned copy of ``doc`` (docs/01 §4). Code blocks are never altered
    except for Unicode normalisation of line endings; table text is whitespace-normalised."""
    # 1) unicode normalisation (code: line endings only)
    blocks: list[Block] = []
    for b in doc.blocks:
        if b.type == BlockType.CODE:
            text = b.text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
            blocks.append(b.model_copy(update={"text": text}))
            continue
        update: dict[str, object] = {"text": normalize_unicode(b.text)}
        if b.caption:
            update["caption"] = normalize_unicode(b.caption).strip()
        blocks.append(b.model_copy(update=update))
    # 2) page furniture must go *before* line merging glues it onto body text
    blocks = _strip_page_furniture(blocks, furniture_threshold)
    # 3) line repair and whitespace
    repaired: list[Block] = []
    for b in blocks:
        if b.type == BlockType.CODE:
            repaired.append(b)
            continue
        text = b.text
        if email and b.type == BlockType.PARAGRAPH:
            text = clean_email(text)
        if b.type in (BlockType.PARAGRAPH, BlockType.LIST):
            text = merge_broken_lines(dehyphenate(text))
        text = normalize_whitespace(text)
        if b.type == BlockType.HEADING:
            text = " ".join(text.split())
        repaired.append(b.model_copy(update={"text": text}))
    blocks = repaired
    blocks = [b for b in blocks if b.text.strip() or b.type in (BlockType.FIGURE,)]
    out = doc.model_copy(update={"blocks": blocks}, deep=True)
    if out.title:
        out.title = " ".join(normalize_unicode(out.title).split())
    if detect_lang:
        lang, conf = detect_language(out.text)
        if lang != "und":
            out.language = lang
        out.metadata["language_confidence"] = conf
        annotate_section_languages(out)
    return out
