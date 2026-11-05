from chatbot.core.types import Chunk, ScoredChunk
from chatbot.generate import (
    CitationVerifier,
    LLMSupportJudge,
    extract_citations,
    parse_markers,
    split_claims,
    strip_dangling,
)
from chatbot.llm import FakeLLM


def sc(i: int, text: str, doc: str = "d1") -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=f"c{i}", doc_id=doc, text=text, title=f"T{i}"), score=1.0 / i
    )


SOURCES = {
    1: sc(1, "Refunds are issued within 14 days of the request."),
    2: sc(2, "Premium plans include 24/7 phone support."),
    3: sc(3, "The warranty lasts 2 years for all hardware."),
}


def test_parse_marker_forms():
    ms = parse_markers("a [1] b [1][3] c [1, 2] d [2-3] e [link](http://x) f [2;3]")
    assert [m.indices for m in ms] == [[1], [1], [3], [1, 2], [2, 3], [2, 3]]


def test_out_of_range_markers_dropped_and_dangling_stripped():
    text = "Refunds take 14 days [1][7]. Support is 24/7 [2, 9]. Bad [8]. Warranty is 2 years [3"
    res = extract_citations(text, SOURCES)
    assert res.invalid_indices == [7, 8, 9]
    assert "[7]" not in res.text and "[8]" not in res.text and "9" not in res.text.split("24/7")[1]
    assert res.text.endswith("Warranty is 2 years")
    assert "Bad." in res.text
    assert [c.index for c in res.citations] == [1, 2]
    assert res.citations[0].chunk_id == "c1" and res.citations[0].snippet.startswith("Refunds")
    assert strip_dangling("foo [1, ") == "foo"


def test_split_claims_attaches_trailing_markers():
    claims = split_claims("Refunds take 14 days. [1] Support is 24/7 [2].")
    assert [(c.text, c.indices) for c in claims] == [
        ("Refunds take 14 days.", [1]),
        ("Support is 24/7.", [2]),
    ]


async def test_lexical_verification_precision_recall():
    text = "Refunds are issued within 14 days [1]. The warranty lasts 5 years [3]. Phone support is 24/7 [1]."
    res = extract_citations(text, SOURCES)
    report = await CitationVerifier().verify(res.text, SOURCES, res.citations)
    assert len(report.claims) == 3
    assert report.claims[0].supported
    assert not report.claims[1].supported  # 5 years is not in the source
    assert not report.claims[2].supported  # cited the wrong source
    assert abs(report.precision - 1 / 3) < 1e-9
    assert abs(report.recall - 1 / 3) < 1e-9
    by_idx = {c.index: c.supported for c in report.citations}
    assert by_idx == {1: True, 3: False}


async def test_llm_judge():
    llm = FakeLLM(handler=lambda r: '{"supported": true, "reason": "ok"}')
    report = await CitationVerifier(LLMSupportJudge(llm)).verify("Anything at all [2].", SOURCES)
    assert report.precision == 1.0 and report.recall == 1.0
    assert len(llm.calls) == 1
    bad = FakeLLM(handler=lambda r: "nope")  # judge failure falls back to lexical
    report = await CitationVerifier(LLMSupportJudge(bad)).verify(
        "Premium plans include phone support [2].", SOURCES
    )
    assert report.recall == 1.0
