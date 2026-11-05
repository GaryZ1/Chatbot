# 01 — Ingestion & Document Parsing

> "Garbage in, garbage retrieved." OHRBench (arXiv 2412.02592, *"OCR Hinders RAG"*) shows OCR and parse errors propagate through **both** retrieval and generation. No downstream trick fully recovers them.

## 1. Connectors and sync

| Concern | What to do |
|---|---|
| **Source systems** | SharePoint, Confluence, Google Drive, Notion, Jira, Zendesk, Salesforce, Slack, Git, S3, SQL |
| **Sync mode** | Full crawl (initial) → incremental (webhooks or change feeds, `modified_since`) → periodic reconciliation (catch deletes) |
| **Identity** | Pull ACLs (users, groups) with every document; sync group membership separately |
| **Provenance** | Store `source_uri`, `doc_id`, `version`, `last_modified`, `author`, `content_hash` |
| **Deletes** | Tombstones. Deleted source documents must disappear from the index within your SLA (a compliance risk otherwise) |
| **Dedup** | Exact (SHA-256 of normalized text) + near-duplicate (MinHash/SimHash, Jaccard > 0.9) |

## 2. Parsing approaches, all of them

| Method | What it is | Purpose | Pros | Cons | Best for |
|---|---|---|---|---|---|
| **Plain text extraction** (pypdf, pdfminer, PyMuPDF) | Reads the PDF text layer | Speed | Very fast, cheap, deterministic | Loses reading order, tables, and headers; fails on scans | Simple, born-digital, single-column text |
| **HTML/Markdown-native** (trafilatura, BeautifulSoup, readability) | DOM-based boilerplate removal | Clean web pages | Keeps headings, lists, and links | Navigation and ad noise if poorly tuned | Websites, wikis, Confluence |
| **Rule/heuristic layout** (Unstructured "fast", pdfplumber) | Coordinates + heuristics for blocks and tables | Basic structure | Cheap; CPU only | Brittle on complex layouts | Forms, invoices with stable templates |
| **Layout-model pipelines** (Docling, Marker, MinerU pipeline, Unstructured "hi_res") | Layout detection → OCR → table structure model (e.g. TableFormer) → reading order | Faithful structure → Markdown/JSON | Strong tables, headings, reading order; open source; runs locally | Slower (GPU helps); multi-model complexity | **Default for mixed enterprise corpora** (Docling), scientific and formula-heavy papers (MinerU) |
| **Classical OCR** (Tesseract, PaddleOCR, cloud OCR) | Image → text | Scanned documents | Mature, many languages | Plain text only; weak layout | Scans, faxes |
| **End-to-end document VLMs** (PaddleOCR-VL, MinerU2.5, olmOCR 2, Qwen-VL / GPT/Claude/Gemini vision, Mistral OCR) | A vision-language model reads the page image and emits Markdown/HTML | Hardest layouts, handwriting, charts | Top accuracy on OmniDocBench-style benchmarks; handles charts and figures | GPU cost; can *hallucinate* text; non-deterministic | Complex scans, charts, mixed-layout reports |
| **Cloud document AI** (Azure Document Intelligence, AWS Textract, Google Document AI, LlamaParse, Reducto) | Managed APIs | Turnkey quality | Good tables and forms, SLAs | $/page, data leaves your boundary | Teams without ML infra |
| **Skip parsing: visual retrieval** (ColPali / ColQwen) | Embed the *page image* directly | Avoid parse loss | No OCR errors; captures charts | Big multi-vector index; still needs a VLM to answer | Slide decks, infographics (see Module 09) |

**2026 landscape notes:**
- **OmniDocBench** is the main end-to-end parsing benchmark. Its 2026 leaderboard is led by document VLMs (e.g. PaddleOCR-VL-1.5 reported around 94.5 overall) *(aggregator figures; verify)*.
- **Docling** (IBM, open source) is widely recommended as the safest default for mixed corpora. **MinerU** is the usual choice for formula-heavy scientific PDFs and CJK text. **olmOCR 2** (Ai2) is fully Apache-2.0 but its activity has reportedly slowed in 2026.
- Post-processing the OCR output measurably improves downstream RAG (MinerU-Popo, 2026).

## 3. Special content types

| Content | Strategy | Pitfall |
|---|---|---|
| **Tables** | Keep each table atomic; serialize as Markdown or HTML (HTML keeps merged cells); also store a *natural-language summary* of the table for embedding; for big tables, chunk by row groups with the header repeated | Flattened tables lose column semantics; numbers retrieved without units |
| **Charts/figures** | VLM caption ("Bar chart: revenue by region 2023–2025; APAC highest at $4.2B") + keep an image reference for multimodal answering | Captions hallucinate numbers; validate on a sample |
| **Slides** | One slide = one unit; include speaker notes; deck title as context | Bullet fragments lack context → enrich |
| **Spreadsheets** | Usually *not* RAG. Route to text-to-SQL or a pandas tool; embed only the schema and descriptions | Embedding raw cells is useless |
| **Code** | AST-aware parsing (tree-sitter); unit = function or class + docstring + file path | Splitting mid-function |
| **Email/tickets** | Strip quoted replies and signatures; thread reconstruction; extract problem → resolution | Duplicated quoted text dominates the index |
| **Transcripts** | Speaker turns + timestamps; topic segmentation | Filler words; ASR errors on jargon |
| **Structured DB rows** | Don't embed them. Use text-to-SQL or API tools; optionally embed "row cards" for entity lookup | Stale copies |

## 4. Cleaning and normalization
- Unicode NFKC normalization; fix hyphenation across line breaks; merge broken lines.
- Remove repeating headers, footers, and page numbers (detect strings that repeat on more than 50% of pages).
- Normalize whitespace; keep Markdown structure (`#`, lists, tables).
- Language detection per document *and* per section (for multilingual routing).
- Detect and **flag** low-quality parses (high OOV ratio, many replacement characters, very low text density). Route those to the VLM fallback.
- PII detection and redaction at ingest when policy requires it (Presidio or cloud DLP).

## 5. Canonical document schema (output of this stage)

```json
{
  "doc_id": "confluence:ENG:123456",
  "version": "v17",
  "source_uri": "https://...",
  "title": "Payment Retry Policy",
  "breadcrumbs": ["Engineering", "Payments", "Policies"],
  "language": "en",
  "created_at": "...", "updated_at": "...", "author": "...",
  "acl": {"allow_groups": ["eng-payments"], "deny_users": []},
  "content_hash": "sha256:...",
  "blocks": [
    {"type": "heading", "level": 1, "text": "Payment Retry Policy"},
    {"type": "paragraph", "text": "..."},
    {"type": "table", "html": "<table>...</table>", "summary": "..."},
    {"type": "figure", "image_ref": "s3://...", "caption": "..."}
  ],
  "parse_quality": {"parser": "docling-2.x", "score": 0.93}
}
```

Keep **blocks with types and hierarchy**. Every later stage (chunking, enrichment, citations) depends on them.

## 6. How to evaluate parsing
- Sample 50–100 pages stratified by format. Have a human mark text, table, and reading-order errors.
- Metrics: character/word edit distance (vs. ground truth), table TEDS score, reading-order accuracy.
- **Downstream test:** run the RAG golden set using parser A vs. parser B. This is the metric that matters.

## Expert checks
- Why can a VLM parser score higher on OmniDocBench yet hurt RAG on your corpus? (Hallucinated text, non-determinism, cost-driven sampling.)
- How do you make sure a document deleted in SharePoint is gone from the index?
- Why is embedding raw spreadsheet cells a mistake, and what do you do instead?
