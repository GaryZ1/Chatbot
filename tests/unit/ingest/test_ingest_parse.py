"""Parsers: Markdown, HTML, text, email, code, PDF and parse-quality scoring."""

from __future__ import annotations

import pytest

from chatbot.core.types import ACL, BlockType
from chatbot.ingest import (
    LOW_QUALITY_THRESHOLD,
    HTMLParser,
    MarkdownParser,
    ParseError,
    SourceRecord,
    default_registry,
    needs_fallback,
    score_parse_quality,
)

MD = """---
title: Payment Retry Policy
doc_type: policy
owner: payments
---
# Payment Retry Policy

Failed payments are retried
three times.

## Schedule

- First retry after 1 hour
- Second retry after 24 hours
  with backoff

| Attempt | Delay |
|---|:---:|
| 1 | 1h |
| 2 | 24h |

```python
def retry():
    return True
```

![Retry flow](s3://bucket/flow.png)

Setext Heading
--------------

Tail paragraph.
"""


def _rec(text: str, uri: str, mime: str = "text/plain", **kw: object) -> SourceRecord:
    return SourceRecord(doc_id="d1", uri=uri, text=text, mime=mime, **kw)  # type: ignore[arg-type]


def test_markdown_blocks() -> None:
    doc = default_registry().parse(
        _rec(MD, "file:///p.md", "text/markdown", acl=ACL(allow_groups=["g"]), tenant_id="t")
    )
    types = [b.type for b in doc.blocks]
    assert types == [
        BlockType.HEADING,
        BlockType.PARAGRAPH,
        BlockType.HEADING,
        BlockType.LIST,
        BlockType.TABLE,
        BlockType.CODE,
        BlockType.FIGURE,
        BlockType.HEADING,
        BlockType.PARAGRAPH,
    ]
    assert doc.title == "Payment Retry Policy"
    assert doc.doc_type == "policy"
    assert doc.metadata["owner"] == "payments"
    assert doc.blocks[1].text == "Failed payments are retried three times."  # broken line merged
    assert doc.blocks[2].level == 2
    assert doc.blocks[3].text.splitlines() == [
        "- First retry after 1 hour",
        "- Second retry after 24 hours with backoff",
    ]
    table = doc.blocks[4]
    assert (
        table.html is not None and "<th>Attempt</th>" in table.html and "<td>24h</td>" in table.html
    )
    assert table.text.splitlines()[0] == "| Attempt | Delay |"
    assert table.metadata["n_rows"] == 2
    code = doc.blocks[5]
    assert code.language == "python" and code.text == "def retry():\n    return True"
    fig = doc.blocks[6]
    assert fig.image_ref == "s3://bucket/flow.png" and fig.caption == "Retry flow"
    assert doc.blocks[7].text == "Setext Heading" and doc.blocks[7].level == 2
    assert doc.acl.allow_groups == ["g"] and doc.tenant_id == "t"
    assert doc.language == "en"
    assert doc.parse_quality.parser == "markdown" and doc.parse_quality.score > 0.9


HTML = """<html><head><title>Site | Retry</title><style>.x{color:red}</style>
<script>var tracking = 1;</script></head>
<body>
<nav><ul><li>Home</li><li>About</li></ul></nav>
<div role="banner">Big banner</div>
<main>
<h1>Retry Policy</h1>
<p>Payments are <b>retried</b> three&nbsp;times.</p>
<h2>Details</h2>
<ol><li>First</li><li>Second<ul><li>Nested</li></ul></li></ol>
<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>
<pre><code class="language-go">fmt.Println("x")
  indented</code></pre>
<img src="chart.png" alt="Revenue chart">
</main>
<aside>Related links</aside>
<footer>Copyright ACME</footer>
</body></html>"""


def test_html_drops_boilerplate_and_keeps_structure() -> None:
    doc = HTMLParser().parse(_rec(HTML, "https://x/retry.html", "text/html"))
    text = doc.text
    for junk in ("Home", "About", "tracking", "Big banner", "Related links", "Copyright", "color"):
        assert junk not in text
    assert doc.title == "Retry Policy"
    kinds = [(b.type, b.level) for b in doc.blocks]
    assert kinds[0] == (BlockType.HEADING, 1)
    assert (BlockType.HEADING, 2) in kinds
    para = next(b for b in doc.blocks if b.type == BlockType.PARAGRAPH)
    assert para.text == "Payments are retried three\xa0times." or para.text.startswith(
        "Payments are retried"
    )
    lst = next(b for b in doc.blocks if b.type == BlockType.LIST)
    assert lst.text.splitlines() == ["1. First", "2. Second", "  - Nested"]
    table = next(b for b in doc.blocks if b.type == BlockType.TABLE)
    assert table.html and "<td>2</td>" in table.html
    code = next(b for b in doc.blocks if b.type == BlockType.CODE)
    assert code.language == "go" and "\n  indented" in code.text
    fig = next(b for b in doc.blocks if b.type == BlockType.FIGURE)
    assert fig.image_ref == "chart.png" and fig.caption == "Revenue chart"


def test_plain_text_and_code_parsers() -> None:
    reg = default_registry()
    doc = reg.parse(_rec("First para.\n\nSecond para\ncontinued.", "file:///n.txt"))
    assert [b.text for b in doc.blocks] == ["First para.", "Second para continued."]
    src = "import os\n\n\ndef f():\n    return os.sep\n"
    code = reg.parse(_rec(src, "file:///m.py", "text/x-python"))
    assert code.doc_type == "code"
    assert len(code.blocks) == 1 and code.blocks[0].type == BlockType.CODE
    assert code.blocks[0].language == "python"
    assert code.blocks[0].text == src.strip("\n")  # code is never re-flowed


def test_email_parser_strips_quotes_and_signature() -> None:
    raw = (
        "From: Alice <alice@example.com>\r\nTo: support@example.com\r\nSubject: Refund for order 42\r\n"
        "Message-ID: <m1@example.com>\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n"
        "Hi team,\r\n\r\nMy refund for order 42 has not arrived.\r\n\r\nBest regards,\r\nAlice\r\n\r\n"
        "On Mon, Jan 1, 2024 at 10:00 AM Bob <bob@example.com> wrote:\r\n> Old quoted text\r\n"
    )
    rec = SourceRecord(
        doc_id="e1", uri="file:///m.eml", content=raw.encode(), mime="message/rfc822"
    )
    doc = default_registry().parse(rec)
    assert doc.title == "Refund for order 42"
    assert doc.doc_type == "email"
    assert doc.metadata["email_from"].startswith("Alice")
    assert "refund for order 42 has not arrived" in doc.text
    assert "Old quoted text" not in doc.text
    assert "Best regards" not in doc.text


def _make_pdf(pages: list[list[str]]) -> bytes:
    """Minimal valid PDF with a Helvetica text layer, one text line per entry."""
    objs: list[bytes] = []
    n_pages = len(pages)
    page_ids = [4 + 2 * i for i in range(n_pages)]
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, lines in enumerate(pages):
        content_id = page_ids[i] + 1
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        ops = ["BT", "/F1 12 Tf", "14 TL", "72 720 Td"]
        for line in lines:
            esc = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"({esc}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode()
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def test_pdf_parser_pages_and_furniture_removal() -> None:
    pytest.importorskip("pypdf")
    body = "Revenue grew in the third quarter because enterprise customers renewed their contracts early."
    pages = [["ACME Confidential", f"Page {i} body: {body}", f"Page {i} of 4"] for i in range(1, 5)]
    rec = SourceRecord(
        doc_id="p1", uri="file:///r.pdf", content=_make_pdf(pages), mime="application/pdf"
    )
    doc = default_registry().parse(rec)
    assert doc.parse_quality.parser.startswith("pypdf")
    assert doc.metadata["page_count"] == 4
    assert {b.page for b in doc.blocks} == {1, 2, 3, 4}
    assert "Revenue grew" in doc.text
    assert "ACME Confidential" not in doc.text  # repeated header removed (>50% of pages)
    assert "of 4" not in doc.text  # page numbers removed


def test_quality_scoring_flags_garbage() -> None:
    good = MarkdownParser().parse(
        _rec("# T\n\n" + "This is a normal sentence about payments. " * 10, "file:///g.md")
    )
    assert score_parse_quality(good).score > 0.9
    garbage = "�� x9#@q Zk$%1 ~~!~ qq� " * 20
    bad = MarkdownParser().parse(_rec(garbage, "file:///b.md"))
    q = score_parse_quality(bad)
    assert q.score < LOW_QUALITY_THRESHOLD
    assert "replacement_chars" in q.flags
    bad.parse_quality = q
    assert needs_fallback(bad)
    empty = MarkdownParser().parse(_rec("", "file:///e.md"))
    assert score_parse_quality(empty).flags == ["empty"]
    sparse = MarkdownParser().parse(_rec("tiny", "file:///s.md"))
    assert "low_text_density" in score_parse_quality(sparse, pages=3).flags


def test_registry_routing_and_tombstones() -> None:
    reg = default_registry()
    with pytest.raises(ParseError):
        reg.parse(
            SourceRecord(doc_id="x", uri="file:///x.bin", content=b"\x00", mime="application/x-foo")
        )
    # unknown text/* falls back to the plain-text parser
    doc = reg.parse(_rec("hello world", "file:///x.weird", "text/x-weird"))
    assert doc.text == "hello world"
    tomb = reg.parse(SourceRecord(doc_id="gone", uri="file:///g.md", deleted=True))
    assert tomb.deleted and tomb.blocks == [] and tomb.doc_id == "gone"


async def test_aparse_runs_in_thread() -> None:
    doc = await default_registry().aparse(
        _rec("# H\n\nBody text here.", "file:///a.md", "text/markdown")
    )
    assert doc.title == "H"
