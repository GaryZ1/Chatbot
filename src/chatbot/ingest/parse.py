"""Parsers: source records → canonical :class:`Document` with typed blocks
(docs/01-ingestion-parsing.md §2, §3 and §5).

Built-in, dependency-free parsers:

* :class:`MarkdownParser` – ATX/setext headings, lists, fenced code, pipe tables
  (``TABLE`` blocks carry both pipe text and HTML), images → ``FIGURE`` blocks,
  YAML front matter → metadata.
* :class:`HTMLParser` – stdlib ``html.parser`` DOM walk that drops boilerplate
  (``nav``/``script``/``style``/``footer``/``aside``/…) and keeps headings, lists,
  ``pre`` code, tables and figures.
* :class:`PlainTextParser` – blank-line paragraphs.
* :class:`EmailParser` – RFC 822 messages via stdlib ``email``.
* :class:`CodeParser` – source files as a single ``CODE`` block (AST chunking later).

Optional (lazy imports): :class:`PDFParser` (``pypdf``; one block group per page
with page numbers) and :class:`DoclingParser` (``docling`` layout-model pipeline).

Every parse is scored by :func:`score_parse_quality` (replacement characters,
text density, OOV-ish ratio). Documents below :data:`LOW_QUALITY_THRESHOLD`
should be routed to a VLM / layout-model fallback.
"""

from __future__ import annotations

import email
import email.policy
import html
import io
import logging
import re
import tempfile
from collections.abc import Iterable, Sequence
from html.parser import HTMLParser as _StdHTMLParser
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

from chatbot.core.types import Block, BlockType, Document, ParseQuality
from chatbot.ingest.clean import clean_document, clean_email
from chatbot.ingest.connectors import SourceRecord

logger = logging.getLogger(__name__)

#: Parse-quality score below which a document should go to the VLM fallback (docs/01 §4).
LOW_QUALITY_THRESHOLD = 0.6


class ParseError(ValueError):
    """The record could not be parsed by any registered parser."""


@runtime_checkable
class Parser(Protocol):
    name: str
    mimes: tuple[str, ...]
    extensions: tuple[str, ...]

    def parse(self, record: SourceRecord) -> Document: ...


# --------------------------------------------------------------------------- helpers


def _base_document(record: SourceRecord, blocks: list[Block], parser: str, **kw: Any) -> Document:
    title = kw.pop("title", "") or _first_heading(blocks) or record.title
    version = str(record.metadata.get("version") or (record.content_hash or "")[7:19] or "1")
    metadata = {**record.metadata, "mime": record.mime, **kw.pop("metadata", {})}
    return Document(
        doc_id=record.doc_id,
        version=version,
        source_uri=record.uri,
        title=title,
        breadcrumbs=list(record.metadata.get("breadcrumbs", [])),
        created_at=record.metadata.get("created_at"),
        updated_at=record.modified,
        author=record.author,
        tenant_id=record.tenant_id,
        acl=record.acl.model_copy(deep=True),
        doc_type=str(record.metadata.get("doc_type", kw.pop("doc_type", "other"))),
        blocks=blocks,
        metadata=metadata,
        parse_quality=ParseQuality(parser=parser),
        **kw,
    )


def _first_heading(blocks: Sequence[Block]) -> str:
    for b in blocks:
        if b.type == BlockType.HEADING and b.text.strip():
            return b.text.strip()
    return ""


def table_rows_to_markdown(rows: Sequence[Sequence[str]]) -> str:
    """Serialise rows (first row = header) as a pipe table."""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [[c.replace("|", "\\|").replace("\n", " ").strip() for c in r] for r in rows]
    norm = [r + [""] * (width - len(r)) for r in norm]
    lines = ["| " + " | ".join(norm[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in norm[1:]]
    return "\n".join(lines)


def table_rows_to_html(rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "<table></table>"
    head = "".join(f"<th>{html.escape(c.strip())}</th>" for c in rows[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(c.strip())}</td>" for c in r) + "</tr>"
        for r in rows[1:]
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _split_pipe_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells = re.split(r"(?<!\\)\|", s)
    return [c.strip().replace("\\|", "|") for c in cells]


_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")


# --------------------------------------------------------------------------- quality


_WORDISH = re.compile(r"\S+")
_PLAUSIBLE_WORD = re.compile(
    r"""^[(\["'“‘]*(?:
        [^\W\d_]+(?:['’\-][^\W\d_]+)*     # word, contractions, hyphenation
      | \d+(?:[.,:/\-]\d+)*%?             # numbers, dates, times
      | [A-Z0-9]+(?:[\-_./][A-Z0-9]+)+    # identifiers like SKU-123
      | \S{1}                             # single symbol
      | https?://\S+ | \S+@\S+            # urls, emails
      | [#*\-+>|`~=:]+                    # markdown punctuation
    )[)\]"'”’.,;:!?…]*$""",
    re.VERBOSE,
)


def score_parse_quality(
    doc: Document, *, raw_size: int | None = None, pages: int | None = None
) -> ParseQuality:
    """Heuristic parse-quality score in [0, 1] with diagnostic flags (docs/01 §4).

    Penalties:
      * ``replacement_chars`` – U+FFFD / mojibake share (broken encodings or fonts);
      * ``low_text_density`` – very little text per page (scans without a text layer)
        or per input byte;
      * ``high_oov`` – share of tokens that do not look like words, numbers or
        identifiers (garbled OCR / font-encoding issues);
      * ``empty`` – nothing extracted.
    """
    text = doc.text
    flags: list[str] = []
    if not text.strip():
        return ParseQuality(parser=doc.parse_quality.parser, score=0.0, flags=["empty"])
    score = 1.0
    n = len(text)
    bad = text.count("\ufffd") + len(re.findall(r"Ã.|â€", text))
    repl_ratio = bad / n
    if repl_ratio > 0.001:
        flags.append("replacement_chars")
        score -= min(0.6, repl_ratio * 30)
    if pages:
        per_page = len(text.strip()) / pages
        if per_page < 200:
            flags.append("low_text_density")
            score -= 0.5 * (1 - per_page / 200)
    elif raw_size and raw_size > 2048:
        density = len(text.strip()) / raw_size
        if density < 0.02:
            flags.append("low_text_density")
            score -= 0.3
    tokens = _WORDISH.findall(text[:50000])
    prose_tokens = [t for t in tokens if len(t) < 40]
    if len(prose_tokens) >= 20:
        oov = sum(1 for t in prose_tokens if not _PLAUSIBLE_WORD.match(t)) / len(prose_tokens)
        if oov > 0.15:
            flags.append("high_oov")
            score -= min(0.5, (oov - 0.15) * 2)
    return ParseQuality(
        parser=doc.parse_quality.parser, score=round(max(0.0, min(1.0, score)), 3), flags=flags
    )


def needs_fallback(doc: Document, threshold: float = LOW_QUALITY_THRESHOLD) -> bool:
    """True when the parse should be retried with a VLM / layout-model parser."""
    return doc.parse_quality.score < threshold


# --------------------------------------------------------------------------- markdown

_ATX = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([\w+\-.#]*)")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+")
_IMAGE_LINE = re.compile(r"^\s*!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)\s*$")


def parse_markdown_blocks(
    text: str, *, page: int | None = None
) -> tuple[list[Block], dict[str, Any]]:
    """Parse Markdown into typed blocks. Returns ``(blocks, front_matter)``."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    front: dict[str, Any] = {}
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            try:
                loaded = yaml.safe_load(text[4:end])
                if isinstance(loaded, dict):
                    front = loaded
                    text = text[end + 4 :].lstrip("\n")
            except yaml.YAMLError:
                logger.debug("ignoring invalid YAML front matter")
    lines = text.split("\n")
    blocks: list[Block] = []
    para: list[str] = []

    def flush_para() -> None:
        if para:
            body = "\n".join(para).strip()
            if body:
                kind = BlockType.LIST if _LIST_ITEM.match(para[0]) else BlockType.PARAGRAPH
                blocks.append(Block(type=kind, text=body, page=page))
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        fence = _FENCE.match(line)
        if fence:
            flush_para()
            marker, lang = fence.group(2), fence.group(3) or None
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(marker[0] * len(marker)):
                body.append(lines[i])
                i += 1
            i += 1  # closing fence (or EOF)
            blocks.append(
                Block(type=BlockType.CODE, text="\n".join(body), language=lang, page=page)
            )
            continue
        atx = _ATX.match(line)
        if atx:
            flush_para()
            blocks.append(
                Block(type=BlockType.HEADING, level=len(atx.group(1)), text=atx.group(2), page=page)
            )
            i += 1
            continue
        if (
            para
            and len(para) == 1
            and not _LIST_ITEM.match(para[0])
            and re.match(r"^\s*(=+|-+)\s*$", line)
            and stripped
        ):
            level = 1 if stripped.startswith("=") else 2
            heading = para.pop().strip()
            flush_para()
            blocks.append(Block(type=BlockType.HEADING, level=level, text=heading, page=page))
            i += 1
            continue
        if "|" in line and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]):
            flush_para()
            rows = [_split_pipe_row(line)]
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(_split_pipe_row(lines[i]))
                i += 1
            blocks.append(
                Block(
                    type=BlockType.TABLE,
                    text=table_rows_to_markdown(rows),
                    html=table_rows_to_html(rows),
                    page=page,
                    metadata={"n_rows": len(rows) - 1, "n_cols": len(rows[0])},
                )
            )
            continue
        img = _IMAGE_LINE.match(line)
        if img:
            flush_para()
            alt = img.group(1).strip()
            blocks.append(
                Block(
                    type=BlockType.FIGURE,
                    text=alt,
                    caption=alt or None,
                    image_ref=img.group(2),
                    page=page,
                )
            )
            i += 1
            continue
        if not stripped:
            flush_para()
            i += 1
            continue
        # a list starting right after a paragraph line begins a new block
        if para and _LIST_ITEM.match(line) and not _LIST_ITEM.match(para[0]):
            flush_para()
        para.append(line.rstrip())
        i += 1
    flush_para()
    return blocks, front


class MarkdownParser:
    name = "markdown"
    mimes: tuple[str, ...] = ("text/markdown", "text/x-markdown")
    extensions: tuple[str, ...] = (".md", ".markdown", ".mdx")

    def parse(self, record: SourceRecord) -> Document:
        blocks, front = parse_markdown_blocks(record.as_text())
        extra: dict[str, Any] = {}
        meta = {k: v for k, v in front.items() if isinstance(k, str)}
        if isinstance(front.get("title"), str):
            extra["title"] = front["title"]
        if isinstance(front.get("doc_type"), str):
            extra["doc_type"] = front["doc_type"]
        doc = _base_document(record, blocks, self.name, metadata=meta, **extra)
        if isinstance(front.get("language"), str):
            doc.language = front["language"]
        return doc


# --------------------------------------------------------------------------- html

_SKIP_TAGS = frozenset(
    {
        "script",
        "style",
        "nav",
        "footer",
        "aside",
        "noscript",
        "template",
        "iframe",
        "svg",
        "form",
        "button",
        "select",
        "head",
    }
)
_SKIP_ROLES = frozenset({"navigation", "banner", "contentinfo", "search", "complementary"})
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "main",
        "header",
        "blockquote",
        "dd",
        "dt",
        "br",
        "hr",
        "figcaption",
        "address",
        "body",
        "html",
    }
)
_VOID = frozenset({"br", "hr", "img", "meta", "link", "input", "source", "wbr", "area", "col"})


class _DOMWalker(_StdHTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self.title = ""
        self._buf: list[str] = []
        self._skip_depth = 0
        self._stack: list[str] = []
        self._in_title = False
        self._heading: int | None = None
        self._pre_depth = 0
        self._pre_lang: str | None = None
        self._list_items: list[str] = []
        self._list_depth = 0
        self._in_li = False
        self._ol_counter: list[int] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._table_depth = 0

    # -- utilities
    def _text(self) -> str:
        raw = "".join(self._buf)
        self._buf.clear()
        return raw if self._pre_depth else " ".join(raw.split())

    def _flush(self, kind: BlockType = BlockType.PARAGRAPH) -> None:
        if self._in_li:
            return
        text = self._text()
        if text.strip():
            self.blocks.append(Block(type=kind, text=text))

    def _flush_list(self) -> None:
        if self._list_items:
            self.blocks.append(Block(type=BlockType.LIST, text="\n".join(self._list_items)))
            self._list_items = []

    # -- parser callbacks
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: (v or "") for k, v in attrs}
        if self._skip_depth:
            if tag not in _VOID:
                self._skip_depth += 1
            return
        if (
            tag in _SKIP_TAGS
            or a.get("role", "").lower() in _SKIP_ROLES
            or a.get("aria-hidden") == "true"
        ):
            if tag == "head":
                # still collect <title> from the head
                self._stack.append(tag)
                return
            if tag not in _VOID:
                self._skip_depth = 1
            return
        if tag == "title":
            self._in_title = True
            return
        if tag not in _VOID:
            self._stack.append(tag)
        if self._table is not None:
            if tag == "tr":
                self._row = []
            elif tag in ("td", "th"):
                self._cell = []
            elif tag == "table":
                self._table_depth += 1
            return
        if re.fullmatch(r"h[1-6]", tag):
            self._flush()
            self._flush_list()
            self._heading = int(tag[1])
        elif tag == "pre":
            self._flush()
            self._pre_depth += 1
            self._pre_lang = None
        elif tag == "code" and self._pre_depth:
            m = re.search(r"(?:language|lang)-([\w+#\-]+)", a.get("class", ""))
            if m:
                self._pre_lang = m.group(1)
        elif tag in ("ul", "ol"):
            if self._list_depth == 0:
                self._flush()
            elif self._in_li:
                self._commit_li()
            self._list_depth += 1
            self._ol_counter.append(0 if tag == "ol" else -1)
        elif tag == "li":
            self._commit_li()
            self._in_li = True
        elif tag == "table":
            self._flush()
            self._flush_list()
            self._table = []
            self._table_depth = 1
        elif tag == "img":
            alt = a.get("alt", "").strip()
            src = a.get("src", "")
            self._flush()
            self.blocks.append(
                Block(type=BlockType.FIGURE, text=alt, caption=alt or None, image_ref=src or None)
            )
        elif tag == "br":
            self._buf.append("\n" if self._pre_depth else " ")
        elif tag in _BLOCK_TAGS and not self._in_li:
            self._flush()

    def _commit_li(self) -> None:
        if not self._in_li:
            return
        text = " ".join("".join(self._buf).split())
        self._buf.clear()
        if text:
            indent = "  " * max(0, self._list_depth - 1)
            counter = self._ol_counter[-1] if self._ol_counter else -1
            if counter >= 0:
                self._ol_counter[-1] += 1
                marker = f"{counter + 1}."
            else:
                marker = "-"
            self._list_items.append(f"{indent}{marker} {text}")
        self._in_li = False

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in self._stack:
            while self._stack and self._stack.pop() != tag:
                pass
        if self._table is not None:
            if tag in ("td", "th") and self._cell is not None and self._row is not None:
                self._row.append(" ".join("".join(self._cell).split()))
                self._cell = None
            elif tag == "tr" and self._row is not None:
                if any(c for c in self._row):
                    self._table.append(self._row)
                self._row = None
            elif tag == "table":
                self._table_depth -= 1
                if self._table_depth == 0:
                    rows = self._table
                    self._table = None
                    if rows:
                        self.blocks.append(
                            Block(
                                type=BlockType.TABLE,
                                text=table_rows_to_markdown(rows),
                                html=table_rows_to_html(rows),
                                metadata={"n_rows": len(rows) - 1, "n_cols": len(rows[0])},
                            )
                        )
            return
        if re.fullmatch(r"h[1-6]", tag) and self._heading is not None:
            text = " ".join("".join(self._buf).split())
            self._buf.clear()
            if text:
                self.blocks.append(Block(type=BlockType.HEADING, level=self._heading, text=text))
            self._heading = None
        elif tag == "pre" and self._pre_depth:
            text = "".join(self._buf).strip("\n")
            self._buf.clear()
            self._pre_depth -= 1
            if text.strip():
                self.blocks.append(Block(type=BlockType.CODE, text=text, language=self._pre_lang))
        elif tag == "li":
            self._commit_li()
        elif tag in ("ul", "ol"):
            self._commit_li()
            self._list_depth = max(0, self._list_depth - 1)
            if self._ol_counter:
                self._ol_counter.pop()
            if self._list_depth == 0:
                self._flush_list()
        elif tag in _BLOCK_TAGS and not self._in_li:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
            return
        if self._table is not None:
            if self._cell is not None:
                self._cell.append(data)
            return
        self._buf.append(data)

    def close(self) -> None:
        super().close()
        self._commit_li()
        self._flush_list()
        self._flush()


def parse_html_blocks(markup: str) -> tuple[list[Block], str]:
    """Parse HTML into typed blocks with boilerplate removed. Returns ``(blocks, <title>)``."""
    walker = _DOMWalker()
    walker.feed(markup)
    walker.close()
    return walker.blocks, " ".join(walker.title.split())


class HTMLParser:
    name = "html"
    mimes: tuple[str, ...] = ("text/html", "application/xhtml+xml")
    extensions: tuple[str, ...] = (".html", ".htm", ".xhtml")

    def parse(self, record: SourceRecord) -> Document:
        blocks, title = parse_html_blocks(record.as_text())
        heading = _first_heading(blocks)
        return _base_document(record, blocks, self.name, title=heading or title)


# --------------------------------------------------------------------------- text / email / code


class PlainTextParser:
    name = "text"
    mimes: tuple[str, ...] = ("text/plain", "text/x-rst", "text/csv")
    extensions: tuple[str, ...] = (".txt", ".text", ".log", ".rst")

    def parse(self, record: SourceRecord) -> Document:
        text = record.as_text().replace("\r\n", "\n")
        blocks = [
            Block(
                type=BlockType.LIST if _LIST_ITEM.match(p) else BlockType.PARAGRAPH, text=p.strip()
            )
            for p in re.split(r"\n\s*\n", text)
            if p.strip()
        ]
        return _base_document(record, blocks, self.name)


class EmailParser:
    """RFC 822 email (``.eml``). Quoted replies/signatures are stripped by :func:`clean_document`
    when the registry parses with ``email=True`` (automatic for this parser)."""

    name = "email"
    mimes: tuple[str, ...] = ("message/rfc822",)
    extensions: tuple[str, ...] = (".eml",)

    def parse(self, record: SourceRecord) -> Document:
        msg = email.message_from_bytes(record.as_bytes(), policy=email.policy.default)
        body_part = msg.get_body(preferencelist=("plain", "html"))
        body = body_part.get_content() if body_part is not None else ""
        if body_part is not None and body_part.get_content_type() == "text/html":
            blocks, _ = parse_html_blocks(body)
        else:
            # strip quoted replies/signatures on the whole body, before paragraph splitting
            body = clean_email(body)
            blocks = [
                Block(type=BlockType.PARAGRAPH, text=p.strip())
                for p in re.split(r"\n\s*\n", body)
                if p.strip()
            ]
        subject = str(msg.get("subject", "") or "")
        meta = {
            "email_from": str(msg.get("from", "") or ""),
            "email_to": str(msg.get("to", "") or ""),
            "email_date": str(msg.get("date", "") or ""),
            "message_id": str(msg.get("message-id", "") or ""),
            "in_reply_to": str(msg.get("in-reply-to", "") or ""),
        }
        doc = _base_document(
            record, blocks, self.name, title=subject, metadata=meta, doc_type="email"
        )
        doc.author = doc.author or meta["email_from"] or None
        return doc


_CODE_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "bash",
    ".sql": "sql",
}


class CodeParser:
    """Source files become one ``CODE`` block; :mod:`chatbot.chunking.code_ast` splits them."""

    name = "code"
    mimes: tuple[str, ...] = (
        "text/x-python",
        "text/javascript",
        "text/x-typescript",
        "text/x-go",
        "text/x-rust",
        "text/x-java",
    )
    extensions: tuple[str, ...] = tuple(_CODE_LANG)

    def parse(self, record: SourceRecord) -> Document:
        lang = _CODE_LANG.get(record.extension)
        source = record.as_text().replace("\r\n", "\n").strip("\n")
        blocks = [Block(type=BlockType.CODE, text=source, language=lang)] if source.strip() else []
        doc = _base_document(
            record, blocks, self.name, title=record.title or Path(record.uri).name, doc_type="code"
        )
        doc.metadata["code_language"] = lang
        return doc


# --------------------------------------------------------------------------- pdf / docling (optional)


class PDFParser:
    """Text-layer extraction with ``pypdf`` (``pip install chatbot[parsing]``).

    Fast and deterministic but loses tables/reading order and fails on scans
    (docs/01 §2) — the quality score flags those for the fallback parser.
    """

    name = "pypdf"
    mimes: tuple[str, ...] = ("application/pdf",)
    extensions: tuple[str, ...] = (".pdf",)

    def parse(self, record: SourceRecord) -> Document:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ParseError("PDF parsing requires 'pypdf' (pip install chatbot[parsing])") from exc
        reader = PdfReader(io.BytesIO(record.as_bytes()), strict=False)
        blocks: list[Block] = []
        for page_no, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # pypdf raises many types on damaged pages
                logger.warning("pypdf failed on %s page %d: %s", record.uri, page_no, exc)
                text = ""
            for para in re.split(r"\n\s*\n", text):
                if para.strip():
                    blocks.append(Block(type=BlockType.PARAGRAPH, text=para.strip(), page=page_no))
        title = ""
        try:
            if reader.metadata and reader.metadata.title:
                title = str(reader.metadata.title)
        except Exception:  # metadata decoding errors are not fatal
            title = ""
        doc = _base_document(record, blocks, f"pypdf-{_pypdf_version()}", title=title)
        doc.metadata["page_count"] = len(reader.pages)
        return doc


def _pypdf_version() -> str:
    try:
        import pypdf

        return str(pypdf.__version__)
    except ImportError:  # pragma: no cover
        return "unknown"


class DoclingParser:
    """Layout-model pipeline (Docling) – the recommended default for mixed corpora (docs/01 §2).

    Converts to Markdown with Docling, then reuses :func:`parse_markdown_blocks`
    so the block model is identical to the native Markdown parser.
    """

    name = "docling"
    mimes: tuple[str, ...] = (
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )
    extensions: tuple[str, ...] = (".pdf", ".docx", ".pptx")

    def __init__(self) -> None:
        self._converter: Any = None

    def _get_converter(self) -> Any:
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
            except ImportError as exc:
                raise ParseError("DoclingParser requires 'docling' (pip install docling)") from exc
            self._converter = DocumentConverter()
        return self._converter

    def parse(self, record: SourceRecord) -> Document:
        converter = self._get_converter()
        suffix = record.extension or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix) as fh:
            fh.write(record.as_bytes())
            fh.flush()
            result = converter.convert(fh.name)
        markdown = result.document.export_to_markdown()
        blocks, _ = parse_markdown_blocks(markdown)
        return _base_document(record, blocks, "docling-2.x")


# --------------------------------------------------------------------------- registry


class ParserRegistry:
    """Maps MIME types and extensions to parsers; parses, cleans and scores records."""

    def __init__(self, parsers: Iterable[Parser] = ()) -> None:
        self._by_mime: dict[str, Parser] = {}
        self._by_ext: dict[str, Parser] = {}
        for p in parsers:
            self.register(p)

    def register(
        self,
        parser: Parser,
        *,
        mimes: Sequence[str] | None = None,
        extensions: Sequence[str] | None = None,
    ) -> None:
        """Register ``parser``; later registrations override earlier ones."""
        for m in mimes if mimes is not None else parser.mimes:
            self._by_mime[m.lower()] = parser
        for e in extensions if extensions is not None else parser.extensions:
            self._by_ext[e.lower() if e.startswith(".") else "." + e.lower()] = parser

    def for_record(self, record: SourceRecord) -> Parser:
        parser = self._by_ext.get(record.extension) or self._by_mime.get(record.mime.lower())
        if parser is None and record.mime.startswith("text/"):
            parser = self._by_mime.get("text/plain")
        if parser is None:
            raise ParseError(f"no parser for {record.uri!r} (mime={record.mime})")
        return parser

    def parse(self, record: SourceRecord, *, clean: bool = True) -> Document:
        """Parse → clean → score. Tombstones become ``Document(deleted=True)`` with no blocks."""
        if record.deleted:
            return Document(
                doc_id=record.doc_id,
                source_uri=record.uri,
                tenant_id=record.tenant_id,
                deleted=True,
                metadata=dict(record.metadata),
            )
        parser = self.for_record(record)
        doc = parser.parse(record)
        if clean:
            doc = clean_document(doc, email=parser.name == "email")
        doc.parse_quality = score_parse_quality(
            doc,
            raw_size=len(record.content) if record.content is not None else None,
            pages=doc.metadata.get("page_count")
            if isinstance(doc.metadata.get("page_count"), int)
            else None,
        ).model_copy(update={"parser": doc.parse_quality.parser})
        if needs_fallback(doc):
            logger.info(
                "low parse quality for %s (%.2f, %s) – route to fallback parser",
                record.uri,
                doc.parse_quality.score,
                doc.parse_quality.flags,
            )
        return doc

    async def aparse(self, record: SourceRecord, *, clean: bool = True) -> Document:
        """Parse in a worker thread (PDF/layout parsing is CPU-bound)."""
        import asyncio

        return await asyncio.to_thread(self.parse, record, clean=clean)


def default_registry(*, use_docling: bool = False) -> ParserRegistry:
    """Registry with every built-in parser; ``use_docling`` prefers Docling for PDF/Office."""
    reg = ParserRegistry(
        [
            PlainTextParser(),
            MarkdownParser(),
            HTMLParser(),
            EmailParser(),
            CodeParser(),
            PDFParser(),
        ]
    )
    if use_docling:
        reg.register(DoclingParser())
    return reg
