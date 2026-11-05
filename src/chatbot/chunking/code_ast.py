"""Code-aware chunking (docs/02-chunking.md §2.17, docs/01 §3 "Code").

Unit = function / class / method + file path + imports (+ the enclosing class
signature for methods). Python is parsed with the stdlib :mod:`ast`; oversized
classes are split by method, functions are never split. Other languages use a
regex fallback that cuts at top-level definition lines (with leading comments).
A tree-sitter implementation can replace :func:`regex_definitions` per language.

Chunk ``text`` is the exact source span (offsets map back into ``Document.text``);
``embed_text`` adds ``File: path``, the language, imports and the class header.
Documents without ``CODE`` blocks are delegated to ``fallback`` (structure-aware).
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from urllib.parse import urlparse

from chatbot.chunking.base import BaseChunker, BlockSpan, DocIndex, trim_span
from chatbot.chunking.recursive import RecursiveSplitter
from chatbot.chunking.structure import StructureChunker
from chatbot.core.interfaces import Chunker
from chatbot.core.tokens import Tokenizer
from chatbot.core.types import BlockType, Chunk, Document

logger = logging.getLogger(__name__)

_EXT_LANG = {
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
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
}

_DEF_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"^(?:export\s+(?:default\s+)?)?(?:async\s+)?function\*?\s+(?P<name>[\w$]+)",
        r"^(?:export\s+(?:default\s+)?)?(?:abstract\s+)?class\s+(?P<name>[\w$]+)",
        r"^(?:export\s+)?(?:const|let|var)\s+(?P<name>[\w$]+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[\w$]+)\s*=>",
        r"^(?:export\s+)?(?:interface|type|enum)\s+(?P<name>\w+)",
        r"^func\s+(?:\([^)]*\)\s*)?(?P<name>\w+)",
        r"^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(?P<name>\w+)",
        r"^(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|impl(?:<[^>]*>)?|mod)\s+(?P<name>[\w:<>]+)",
        r"^(?:public|private|protected|internal)?\s*(?:static\s+|final\s+|abstract\s+|sealed\s+)*(?:class|interface|record|enum)\s+(?P<name>\w+)",
        r"^(?:def|class|module)\s+(?P<name>[\w.?!]+)",
    )
)
_COMMENT_LINE = re.compile(r"^\s*(?://|#|/\*|\*|--|;)")
_IMPORT_LINE = re.compile(
    r"^\s*(?:import\s|from\s+\S+\s+import\s|#include\s|using\s+[\w.]+;|use\s+[\w:]+|require\(|package\s)"
)


@dataclass(slots=True)
class CodeUnit:
    """A definition located by line numbers (1-based, inclusive)."""

    start_line: int
    end_line: int
    symbol: str
    symbol_type: str  # function | class | method | module | class_header | definition
    parent_symbol: str | None = None
    header: str = ""  # enclosing class signature for methods
    children: list[CodeUnit] = field(default_factory=list)


def _line_offsets(source: str) -> list[int]:
    offs = [0]
    for i, ch in enumerate(source):
        if ch == "\n":
            offs.append(i + 1)
    return offs


def python_definitions(source: str) -> tuple[list[CodeUnit], list[str]]:
    """Top-level defs/classes (+ methods as children) and import lines via :mod:`ast`.

    Raises :class:`SyntaxError` for invalid Python (callers fall back to regex).
    """
    tree = ast.parse(source)
    lines = source.split("\n")
    imports: list[str] = []
    units: list[CodeUnit] = []
    pending_module: list[int] = []

    def flush_module() -> None:
        if pending_module:
            units.append(CodeUnit(min(pending_module), max(pending_module), "<module>", "module"))
            pending_module.clear()

    def start_of(node: ast.stmt) -> int:
        decos = getattr(node, "decorator_list", [])
        first = min([node.lineno, *(d.lineno for d in decos)])
        while first > 1 and lines[first - 2].strip().startswith("#"):
            first -= 1  # attach leading comments to the definition
        return first

    for node in tree.body:
        end = node.end_lineno or node.lineno
        if isinstance(node, ast.Import | ast.ImportFrom):
            imports.extend(lines[node.lineno - 1 : end])
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            flush_module()
            units.append(CodeUnit(start_of(node), end, node.name, "function"))
        elif isinstance(node, ast.ClassDef):
            flush_module()
            cls = CodeUnit(start_of(node), end, node.name, "class")
            header_line = lines[node.lineno - 1].strip()
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef):
                    cls.children.append(
                        CodeUnit(
                            start_of(sub),
                            sub.end_lineno or sub.lineno,
                            f"{node.name}.{sub.name}",
                            "method",
                            parent_symbol=node.name,
                            header=header_line,
                        )
                    )
            units.append(cls)
        else:
            pending_module.extend(range(start_of(node), end + 1))
    flush_module()
    return units, imports


def regex_definitions(source: str) -> tuple[list[CodeUnit], list[str]]:
    """Language-agnostic fallback: cut at top-level definition lines (column 0)."""
    lines = source.split("\n")
    imports = [ln for ln in lines if _IMPORT_LINE.match(ln)]
    starts: list[tuple[int, str]] = []
    for i, ln in enumerate(lines):
        if not ln or ln[0].isspace():
            continue
        for pat in _DEF_PATTERNS:
            m = pat.match(ln)
            if m:
                begin = i
                while (
                    begin > 0 and lines[begin - 1].strip() and _COMMENT_LINE.match(lines[begin - 1])
                ):
                    begin -= 1
                starts.append((begin, m.group("name")))
                break
    units: list[CodeUnit] = []
    if not starts:
        return [CodeUnit(1, len(lines), "<module>", "module")], imports
    if starts[0][0] > 0:
        units.append(CodeUnit(1, starts[0][0], "<module>", "module"))
    for k, (begin, name) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        units.append(CodeUnit(begin + 1, end, name, "definition"))
    return units, imports


class CodeChunker(BaseChunker):
    """AST/regex code chunker.

    Args:
        max_tokens: classes above this are split by method; module-level code above it
            is recursively split on blank lines. Functions are never split (flagged ``oversize``).
        max_import_lines: imports copied into ``embed_text``.
        fallback: chunker for documents without code blocks.
    """

    name = "code"

    def __init__(
        self,
        max_tokens: int = 512,
        tokenizer: Tokenizer | None = None,
        *,
        max_import_lines: int = 30,
        fallback: Chunker | None = None,
    ) -> None:
        super().__init__(tokenizer, breadcrumbs=False)
        self._validate_sizes(max_tokens)
        self.max_tokens = max_tokens
        self.max_import_lines = max_import_lines
        self.fallback: Chunker = fallback or StructureChunker(max_tokens, tokenizer=self.tokenizer)

    @staticmethod
    def file_path(doc: Document) -> str:
        path = doc.metadata.get("path")
        if isinstance(path, str) and path:
            return path
        if doc.source_uri:
            parsed = urlparse(doc.source_uri)
            return parsed.path or doc.source_uri
        return doc.title or doc.doc_id

    def _language(self, doc: Document, span: BlockSpan) -> str | None:
        if span.block.language:
            return span.block.language.lower()
        lang = doc.metadata.get("code_language")
        if isinstance(lang, str):
            return lang
        return _EXT_LANG.get(PurePosixPath(self.file_path(doc)).suffix.lower())

    def chunk(self, doc: Document) -> list[Chunk]:
        index = DocIndex(doc)
        code_spans = [s for s in index.spans if s.block.type == BlockType.CODE]
        if not code_spans:
            return self.fallback.chunk(doc)
        out: list[Chunk] = []
        for span in code_spans:
            out.extend(self._chunk_block(index, span, start_position=len(out)))
        return out

    def _chunk_block(self, index: DocIndex, span: BlockSpan, start_position: int) -> list[Chunk]:
        doc = index.doc
        source = span.block.text
        lang = self._language(doc, span)
        units: list[CodeUnit]
        imports: list[str]
        if lang in ("python", "py"):
            try:
                units, imports = python_definitions(source)
                parser = "ast"
            except SyntaxError as exc:
                logger.info("python AST failed for %s (%s); regex fallback", doc.doc_id, exc)
                units, imports = regex_definitions(source)
                parser = "regex"
        else:
            units, imports = regex_definitions(source)
            parser = "regex"
        line_offs = _line_offsets(source)
        path = self.file_path(doc)

        def span_of(u: CodeUnit) -> tuple[int, int]:
            s = line_offs[u.start_line - 1]
            e = line_offs[u.end_line] - 1 if u.end_line < len(line_offs) else len(source)
            return trim_span(source, s, e)

        flat: list[CodeUnit] = []
        for u in units:
            s, e = span_of(u)
            if (
                u.symbol_type == "class"
                and u.children
                and self.count(source[s:e]) > self.max_tokens
            ):
                first = u.children[0].start_line
                if first > u.start_line:
                    flat.append(CodeUnit(u.start_line, first - 1, u.symbol, "class_header"))
                for k, m in enumerate(u.children):
                    end_line = (
                        u.children[k + 1].start_line - 1 if k + 1 < len(u.children) else u.end_line
                    )
                    flat.append(
                        CodeUnit(
                            m.start_line, end_line, m.symbol, "method", m.parent_symbol, m.header
                        )
                    )
            else:
                flat.append(u)

        imports_text = "\n".join(imports[: self.max_import_lines])
        splitter = RecursiveSplitter(
            self.max_tokens, separators=("\n\n", "\n", ""), tokenizer=self.tokenizer
        )
        chunks: list[Chunk] = []
        for u in flat:
            s, e = span_of(u)
            if s >= e:
                continue
            ranges = [(s, e)]
            if (
                u.symbol_type in ("module", "definition", "class_header")
                and self.count(source[s:e]) > self.max_tokens
            ):
                ranges = splitter.split(source, s, e)
            for rs, re_ in ranges:
                body = source[rs:re_]
                header = [f"File: {path}"]
                if lang:
                    header.append(f"Language: {lang}")
                if imports_text and u.symbol_type != "module":
                    header.append(imports_text)
                if u.header:
                    header.append(u.header)
                path_parts = [p for p in (u.parent_symbol, u.symbol.split(".")[-1]) if p]
                meta = {
                    "symbol": u.symbol,
                    "symbol_type": u.symbol_type,
                    "code_language": lang,
                    "file_path": path,
                    "start_line": source.count("\n", 0, rs) + 1,
                    "end_line": source.count("\n", 0, re_) + 1,
                    "code_parser": parser,
                    "atomic": True,
                }
                if self.count(body) > self.max_tokens:
                    meta["oversize"] = True
                chunks.append(
                    self.make_chunk(
                        index,
                        position=start_position + len(chunks),
                        start=span.start + rs,
                        end=span.start + re_,
                        heading_path=[*span.heading_path, *path_parts]
                        if u.symbol != "<module>"
                        else list(span.heading_path),
                        embed_text="\n".join(header) + "\n\n" + body,
                        metadata=meta,
                    )
                )
        return chunks


def detect_code_language(path: str) -> str | None:
    """Language from a file extension (helper shared with parsers)."""
    return _EXT_LANG.get(PurePosixPath(path).suffix.lower())
