# pdf2JSON Pipeline

Extracts text from academic PDF/DOCX papers → structured, cleaned JSON (one sentence per row).

## Folder Structure

```
pdf2JSON/
├── input/          # Drop PDF/DOCX files here
├── middle/raw/     # Step 1 output: raw extracted JSON (sections + paragraphs)
├── output/         # Step 2 output: cleaned flat JSON (one sentence per row)
├── main.py         # CLI entry point
├── pipeline.py     # Orchestration: wires extraction → cleaning → flat JSON
├── extraction.py   # Step 1: PDF/DOCX → raw sections
├── cleaning.py     # Step 2: raw text → clean sentences
├── serve.py        # Local server for viewer.html (auto-loads output/ and middle/raw/)
├── viewer.html     # Browser UI to inspect clean vs raw JSON
└── test_parse_7.py # Quick test on 7 edge-case papers
```

## Pipeline Files

### main.py
CLI entry point. Three modes:
- `python main.py` — full pipeline (extract + clean all files in `input/`)
- `python main.py --extract` — step 1 only
- `python main.py --clean` — step 2 only (re-clean from existing `middle/raw/`)
- `python main.py --file paper.pdf` — single file

### pipeline.py
Glue between extraction and cleaning. Key functions:
- `extract_single(filepath)` — calls `extraction.extract_paper()`, wraps result with filename/metadata
- `clean_and_flatten(raw_result)` — calls `cleaning.clean_sections()`, flattens into rows
- `save_json()` / `load_json()` — file I/O helpers

Output row schema:
```json
{
  "file_name": "paper.pdf",
  "format_class": "standard_complete",
  "section_id": 1,
  "section_name": "Introduction",
  "paragraph_id": 0,
  "text": "One complete sentence.",
  "warnings": ""
}
```

### extraction.py
Reads PDF (via PyMuPDF/fitz) or DOCX (via python-docx) into structured sections.

`PDFExtractor`:
- Extracts text spans with font size/bold metadata
- Detects body font size, headers/footers (repeated text on 3+ pages)
- Finds section headers by font size + bold + known header names (KNOWN_HEADERS_RE)
- Stops at References/Appendix sections
- Merges spans into paragraphs using block gaps + sentence boundary detection
- Classifies format: `standard_complete`, `partial_sections`, `no_section_headers`, `abstract_only`, `slides`

`DOCXExtractor`: simpler — uses paragraph styles/bold for section detection.

### cleaning.py
Transforms raw paragraphs into clean, single-sentence chunks.

**Pre-normalization** (`_pre_normalize`):
- Fixes broken PDF diacritics (standalone marks like `¸c` → `ç`, dotless `ı` → `i`)
- NFKC Unicode normalization (composes characters, fixes fullwidth chars, ligatures)
- Dehyphenates line-break hyphens (`institu- tion` → `institution`)

**Cleaning steps** (`clean_paragraph`):
1. Pre-normalize
2. Remove math (LaTeX, broken PDF formulas, Greek-symbol expressions)
3. Remove table content (numeric rows, table headers, figure captions)
4. Remove in-text citations — parenthetical `(Author, Year)` and narrative `Author (Year)`
5. Remove footnote markers
6. Final normalization + artifact cleanup (orphaned symbols, empty brackets, double punctuation)

**Section-level processing** (`clean_sections`):
1. Filter out pure math/table paragraphs
2. Merge broken paragraphs (mid-sentence page/column breaks healed)
3. Clean each merged block
4. Split into individual sentences via `split_sentences()`
5. Drop chunks shorter than 20 characters

### serve.py
Minimal HTTP server for the viewer. Run `python serve.py` → opens browser at `localhost:8787`.
- `/api/files` — lists JSON files in `output/` and `middle/raw/`
- `/api/file/clean/<name>` — serves a cleaned JSON file
- `/api/file/raw/<name>` — serves a raw JSON file
