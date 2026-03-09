# Improvement Plan: Paragraph Filtering & Indexing Pipeline
 
## Design Principle
 
**Filtering/indexing paragraphs is independent of the method taxonomy (L1/L2).**
 
The current pipeline couples paragraph selection to keyword matching — a paragraph
only survives if it contains an L1 keyword. This means the filtering step and the
method classification step are entangled. The improved pipeline separates them:
 
1. **Stage 1 (Method-agnostic):** Filter and index paragraphs based on their
   structural and content properties — section relevance, text quality, minimum
   length. This produces a clean set of "method-relevant chunks" for every paper,
   regardless of any keyword dictionary.
 
2. **Stage 2 (Keyword matching):** Run keyword matching against the filtered chunks
   as a soft signal (hit counts per method), not a hard gate.
 
3. **Stage 3 (Semantic matching):** Embed filtered chunks with SPECTER2 and compute
   similarity against method descriptions using top-k mean aggregation (not max-pool).
 
4. **Stage 4 (Combined scoring):** Merge keyword hits and semantic similarity into a
   final composite score per method per paper.
 
---
 
## What changes
 
### New file: `src/paragraph_filter.py`
Method-agnostic paragraph filtering. Applied to ALL 150 papers.
 
**Logic:**
- **Minimum length filter:** Drop chunks with `len(text) < 100` characters.
  (11.5% of chunks are <50 chars — figure captions, JEL codes, author lines.
  31.5% are <100 chars. These are noise for embedding.)
- **Section-aware relevance scoring:** Classify sections into tiers:
  - **Tier 1 (methodology-relevant):** section names matching patterns like
    `empirical`, `strategy`, `method`, `model`, `identification`, `estimation`,
    `experiment`, `design`, `analysis`, `results`, `data`, `robustness`,
    `mechanism`, `framework`, `approach`.
  - **Tier 2 (contextual):** `introduction`, `background`, `discussion`,
    `conclusion`, `literature`, `related`, `survey`, `review`.
  - **Tier 3 (excluded):** `preamble`, `appendix`, `references`, `proofs`,
    `acknowledgment`, `abstract`, `unknown`.
  - Unrecognized section names default to Tier 1 (assume relevant since many
    papers have specific section titles like "The Impact of Autonomous AI").
- **Output:** For each paper, save all Tier 1 + Tier 2 chunks that pass length
  filter. Each chunk is annotated with its `tier` (1 or 2). Tier 3 chunks are
  excluded entirely.
 
**Output directory:** `filter_chunk/` (replaces the old keyword-dependent output).
Each file is `{paper_id}.json` containing the filtered chunks with a `tier` field
added but NO method-related fields.
 
### Modified file: `src/keyword_matcher.py`
Rewrite keyword matching with these fixes:
 
- **Case-insensitive matching:** Normalize both text and keywords to lowercase.
- **Split compound keywords:** Pre-process keyword strings to extract searchable
  terms. For example:
  - `"Difference-in-Differences (DiD)"` → search for `"difference-in-differences"`
    OR `"did"` (the abbreviation inside parentheses).
  - `"Two-Stage Least Squares (2SLS)"` → search for `"two-stage least squares"`
    OR `"2sls"`.
  - `"Heterogeneity-Robust DiD (e.g., Callaway & Sant'Anna, Sun & Abraham)"`
    → search for `"heterogeneity-robust did"`.
  - General rule: extract the main term (before parentheses) and any abbreviation
    (inside parentheses, excluding "e.g.," prefixes). Both become independent
    search terms.
- **Word-boundary awareness:** Use regex `\b` word boundaries to prevent substring
  matches (e.g., "IV" shouldn't match "prim**iv**e").
  Short abbreviations (<=3 chars like "IV", "RDD", "DiD") get mandatory word
  boundaries. Longer terms use substring matching (they are specific enough).
- **Runs on filtered chunks from Stage 1** (not raw paper chunks).
- **No longer acts as a gate.** Returns keyword hit metadata that gets attached to
  chunks and aggregated per paper, but does NOT discard papers with zero hits.
 
### Modified file: `src/embedder.py`
Minor changes:
 
- Operates on filtered chunks from the new `paragraph_filter.py` output (all 150
  papers, not just keyword-matched ones).
- Skip embedding chunks shorter than 100 chars (should already be filtered, but
  as a safety net).
 
### Modified file: `src/semantic_matcher.py`
Major logic change:
 
- **Replace max-pool with top-k mean aggregation:**
  Instead of `max(similarities)`, compute the mean of the top-k most similar
  chunks (k=3, or fewer if the paper has <3 chunks). This is more robust —
  a single spurious high-similarity chunk can't inflate the score.
- **Tier-weighted scoring:** Tier 1 chunks (methodology sections) get full weight.
  Tier 2 chunks (introduction/conclusion) get 0.5 weight in the aggregation. This
  means a method mentioned only in the introduction scores lower than one discussed
  in the methodology section.
- **Dynamic threshold calibration:** Instead of a fixed 0.5 threshold, compute
  per-paper z-scores: for each paper, calculate mean and std of similarities across
  all methods, then only assign methods that are >1 std above the paper's mean.
  This adapts to each paper's similarity distribution and prevents the "everything
  matches" problem.
 
### Modified file: `src/chunk_filter.py`
Rename/repurpose: now saves the method-agnostic filtered chunks from
`paragraph_filter.py`. Remove the old method-coupled saving logic.
 
### Modified file: `predefine/config.py`
Update configuration:
 
```python
# Paragraph filtering (method-agnostic)
MIN_CHUNK_LENGTH = 100
# Semantic matching
SPECTER2_MODEL_NAME = "allenai/specter2_base"
TOP_K = 3
TIER2_WEIGHT = 0.5
ZSCORE_THRESHOLD = 1.0   # methods must be >1 std above paper mean
# Keyword matching
KEYWORD_BONUS_WEIGHT = 0.1  # bonus added to semantic score per keyword hit
```
 
### Modified file: `main.py`
Restructure the pipeline to reflect the new 4-stage architecture:
 
```
Stage 1: Paragraph Filtering (method-agnostic)
  - Load all 150 papers
  - Apply section-tier classification + min-length filter
  - Save filtered chunks to filter_chunk/
 
Stage 2: Keyword Matching (soft signal)
  - Load L1/L2 keyword definitions
  - Run improved keyword matching on filtered chunks
  - Attach keyword hit metadata to chunks
  - Write keyword_match.csv
 
Stage 3: SPECTER2 Embedding
  - Load model
  - Embed all filtered chunks (all 150 papers)
  - Embed method descriptions
 
Stage 4: Semantic + Combined Matching
  - Run semantic matching with top-k mean + tier weighting + z-score threshold
  - Add keyword bonus to semantic scores
  - Write semantic_matching.csv
```
 
### Modified file: `app.py`
Update Streamlit viewer:
 
- Show ALL 150 papers (not just 38 keyword-matched ones).
- Display tier labels on chunks (Tier 1 / Tier 2).
- Show keyword hits as annotations (like before) but also show papers with
  zero keyword hits.
- Update semantic threshold slider to work with the new z-score approach
  (or keep a manual similarity threshold as an override option).
 
### Files NOT changed
- `src/keyword_loader.py` — works as-is.
- `src/paper_loader.py` — works as-is.
- `predefine/l1_methods.json` — taxonomy stays the same.
- `predefine/l2_methods.json` — taxonomy stays the same.
- `data/` — input data stays the same.
 
---
 
## Summary of architectural changes
 
| Aspect | Current (broken) | Improved |
|--------|-----------------|----------|
| Paragraph filtering | Keyword-dependent (coupled to L1) | Method-agnostic (section tier + length) |
| Papers processed | 38/150 (25.3%) | 150/150 (100%) |
| Keyword role | Hard gate (no keywords = discarded) | Soft signal (bonus to score) |
| Keyword matching | Case-sensitive exact substring | Case-insensitive, split compounds, word boundaries |
| Semantic aggregation | Max-pool (always high) | Top-k mean (k=3) with tier weighting |
| Threshold | Fixed 0.5 (meaningless) | Per-paper z-score (adaptive) |
| Section awareness | None | Tier 1/2/3 classification |
| Junk chunk removal | None | Min 100 chars, exclude Tier 3 sections |
 
---
 
## Implementation order
 
1. `predefine/config.py` — add new config constants
2. `src/paragraph_filter.py` — new file, method-agnostic filtering
3. `src/keyword_matcher.py` — rewrite with case-insensitive + compound splitting
4. `src/chunk_filter.py` — simplify to save method-agnostic chunks
5. `src/semantic_matcher.py` — top-k mean + tier weighting + z-score
6. `src/embedder.py` — minor update (process all papers)
7. `main.py` — restructure pipeline stages
8. `app.py` — update viewer for 150 papers + tier display