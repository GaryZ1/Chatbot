"""Inline the example files into docs/20-advanced-rag-catalog.md.

Each listing in the doc sits between markers that name its source file:

    <!-- code: hyde.py -->
    ...generated, do not edit...
    <!-- /code -->

    python examples/advanced_rag/build_doc.py          # rewrite the doc in place
    python examples/advanced_rag/build_doc.py --check  # exit 1 if the doc is stale (CI)
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOC = HERE.parents[1] / "docs" / "20-advanced-rag-catalog.md"
BLOCK = re.compile(r"(<!-- code: (?P<file>[\w.]+) -->\n)(?P<body>.*?)(<!-- /code -->)", re.S)


def render(doc: str) -> str:
    def repl(m: re.Match) -> str:
        source = (HERE / m.group("file")).read_text().rstrip()
        link = f"[`examples/advanced_rag/{m.group('file')}`](../examples/advanced_rag/{m.group('file')})"
        return f"{m.group(1)}{link}\n\n```python\n{source}\n```\n{m.group(4)}"

    return BLOCK.sub(repl, doc)


def main() -> int:
    current = DOC.read_text()
    fresh = render(current)
    if "--check" in sys.argv:
        if fresh != current:
            print(f"{DOC} is stale: run python examples/advanced_rag/build_doc.py")
            return 1
        return 0
    DOC.write_text(fresh)
    print(f"wrote {DOC} ({len(BLOCK.findall(fresh))} listings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
