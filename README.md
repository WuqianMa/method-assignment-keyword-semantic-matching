# Method Classification Pipeline

Classifies academic papers by research methodology using a four-stage pipeline: **paragraph filtering** → **keyword matching (soft signal)** → **SPECTER2 embedding** → **semantic matching with z-score thresholding**.

Result-preview:

https://method-assignment-keyword-semantic-matching-2gxvkogidmtws73fdb.streamlit.app/

## Project Structure

```
method/
├── main.py                            # Pipeline entry point (4-stage)
├── app.py                             # Streamlit viewer (run separately)
├── requirements.txt                   # Streamlit dependency for app.py
│
├── predefine/                         # Method taxonomy & config
│   ├── l1_methods.json                # 5 L1 categories with keywords
│   ├── l2_methods.json                # 17 L2 sub-methods (linked to L1 via level_1_label)
│   ├── json_describe.md               # Schema docs for the above
│   └── config.py                      # Model name, thresholds, pipeline params
│
├── data/                              # Input data (read-only)
│   ├── metadata/meta_data.json        # 150 papers: paper_id, title, author, etc.
│   ├── papers_json_normalize/*.json   # 150 files, paragraph-level chunks per paper
│   └── data_describe.md
│
├── src/                               # Pipeline modules
│   ├── keyword_loader.py              # Load L1/L2 method definitions
│   ├── paper_loader.py                # Load paper JSONs and metadata
│   ├── paragraph_filter.py            # Method-agnostic section tier + length filtering
│   ├── keyword_matcher.py             # Case-insensitive keyword matching with compound splitting
│   ├── chunk_filter.py                # Save filtered chunks to disk
│   ├── csv_writer.py                  # Write assignment CSVs
│   ├── embedder.py                    # SPECTER2 embedding (all papers)
│   └── semantic_matcher.py            # Top-k mean + tier weighting + z-score matching
│
├── filter_chunk/                      # Output: method-agnostic filtered chunks (all papers)
│   └── {paper_id}.json
├── embedded_chunk/                    # Output: SPECTER2 embeddings of filtered chunks
│   └── {paper_id}.json
├── embedded_method_description/       # Output: SPECTER2 embeddings of method descriptions
│   ├── l1_embeddings.json
│   └── l2_embeddings.json
│
└── assignments/                       # Output: classification results
    ├── keyword_match.csv              # Keyword-based L1/L2 assignments
    ├── semantic_matching.csv          # Semantic-based L1/L2 with similarity scores
    ├── NotebookLM.csv                 # Baseline: NotebookLM RAG assignments
    └── assignment.md
```

## Pipeline Logic

The pipeline runs in four stages:

### Stage 1: Paragraph Filtering (method-agnostic)

Filters all 150 papers based on structural and content properties, independent of any method taxonomy.

- **Minimum length filter:** Drops chunks with `len(text) < 100` characters (removes figure captions, JEL codes, author lines)
- **Section-tier classification:** Classifies sections into three tiers:
  - **Tier 1 (methodology-relevant):** Sections matching patterns like `empirical`, `method`, `model`, `estimation`, `experiment`, `results`, `data`, `robustness`, etc.
  - **Tier 2 (contextual):** `introduction`, `background`, `discussion`, `conclusion`, `literature`, etc.
  - **Tier 3 (excluded):** `preamble`, `appendix`, `references`, `abstract`, `acknowledgment`
  - Unrecognized section names default to **Tier 1** (many papers have paper-specific titles)
- **Output:** Tier 1 + Tier 2 chunks saved to `filter_chunk/{paper_id}.json` with a `tier` field

### Stage 2: Keyword Matching (soft signal)

Runs keyword matching on filtered chunks as a **soft signal** — keyword hits contribute a bonus to semantic scores but do **not** gate papers.

- **Case-insensitive matching** with text and keywords normalized to lowercase
- **Compound keyword splitting:** `"Difference-in-Differences (DiD)"` → searches for `"difference-in-differences"` OR `"did"`; parenthetical abbreviations extracted automatically (excluding `e.g.` prefixes)
- **Word-boundary awareness:** Short abbreviations (≤3 chars like "IV", "DiD") use `\b` regex word boundaries to prevent substring matches. Longer terms use substring matching.
- **No hard gate:** Papers with zero keyword hits are still processed in subsequent stages

### Stage 3: SPECTER2 Embedding

- Embeds **all** filtered chunks from all papers (not just keyword-matched ones)
- Embeds method descriptions (`semantic_meaning` field from L1/L2 JSONs)
- Uses `allenai/specter2_base` with `[CLS]` token → 768-dim vectors
- Safety net: skips chunks shorter than 100 characters

### Stage 4: Semantic + Combined Matching

- **Top-k mean aggregation (k=3):** Instead of `max(similarities)`, computes the mean of the top-k most similar chunks. A single spurious high-similarity chunk can no longer inflate the score.
- **Tier-weighted scoring:** Tier 1 chunks (methodology sections) get full weight (1.0). Tier 2 chunks (introduction/conclusion) get 0.5 weight. A method mentioned only in the introduction scores lower than one discussed in the methodology section.
- **Keyword bonus:** Each keyword hit adds a small bonus (`0.1`) to the semantic similarity score, integrating keyword evidence with semantic matching.
- **Per-paper z-score threshold:** Instead of a fixed threshold, computes z-scores for each paper's similarity distribution. Only assigns methods that are >1 standard deviation above the paper's mean. This adapts to each paper's similarity distribution and prevents the "everything matches" problem.
- **L2 scoping:** L2 methods are only matched against L1 categories that passed the z-score threshold.

## Method Taxonomy

### L1 Categories (5), L2 Sub-methods (17)

| # | Category | L2 Sub-methods |
| --- | --- | --- |
| 1 | Empirical and Econometric Methods | [Panel Data and Fixed Effects, Difference-in-Differences (DiD) & Event Studies, Instrumental Variables (IV) & Shift-Share Designs, Regression Discontinuity Design (RDD), Structural & Advanced Econometrics] |
| 2 | Computational, Data Science, and Machine Learning | [Natural Language Processing (NLP) & Text Analysis, Machine Learning Predictors & Regularization, Network & Spatial Analysis] |
| 3 | Theoretical and Formal Modeling | [Game Theory & Mechanism Design, Behavioral & Cognitive Modeling, Macroeconomic & Spatial Equilibrium] |
| 4 | Experimental Methods | [Field Experiments / RCTs, Laboratory & Online Experiments, Survey & Conjoint Experiments] |
| 5 | Qualitative, Descriptive, and Mixed Methods | [Archival Research & Case Studies, Interviews, Literature Reviews & Aggregation] |

## Usage

### Run the pipeline

```bash
python main.py
```

Requires: `transformers`, `torch`, `numpy`

### Launch the viewer

```bash
streamlit run app.py
```

Requires: `streamlit` (see `requirements.txt`)

The viewer provides:
- **All papers displayed** (not just keyword-matched ones)
- **Tier labels** (Tier 1 / Tier 2) on each filtered chunk
- **Filtered chunk display** with matched keywords highlighted in orange
- **Three assignment columns**: NotebookLM (baseline), Keyword Match, Semantic Match
- **Threshold sliders** in the sidebar to override z-score filtering for display
- **Stats panel** showing total papers, keyword match count, and semantic match count
- L2 methods grouped under their parent L1 in both keyword and semantic displays

### Configuration

Edit `predefine/config.py` to change:
- `SPECTER2_MODEL_NAME` — HuggingFace model identifier (default: `allenai/specter2_base`)
- `MIN_CHUNK_LENGTH` — minimum character length for chunks (default: 100)
- `TOP_K` — number of top chunks for similarity aggregation (default: 3)
- `TIER2_WEIGHT` — weight for Tier 2 chunks in aggregation (default: 0.5)
- `ZSCORE_THRESHOLD` — z-score cutoff for method assignment (default: 1.0)
- `KEYWORD_BONUS_WEIGHT` — bonus per keyword hit added to semantic score (default: 0.1)

## Pipeline Results

On 150 input papers (economics / social science corpus):

### Coverage

| Stage | Result |
| --- | --- |
| Paragraph filtering | 144/150 papers passed (9,697 chunks) |
| Keyword matching | 132/150 papers had keyword hits |
| SPECTER2 embedding | 145 papers embedded |
| Semantic assignment | 145 papers scored |

### L1 Assignment Distribution

| L1 methods assigned | Papers | Share |
| --- | --- | --- |
| 0 | 6 | 4% |
| 1 | 132 | 91% |
| 2 | 7 | 5% |
| 3+ | 0 | 0% |

The z-score threshold eliminates the "everything matches" problem — no paper is assigned more than 2 L1 categories.

### L1 Frequency

| L1 Category | Papers |
| --- | --- |
| Empirical and Econometric Methods | 100 |
| Theoretical and Formal Modeling | 30 |
| Computational, Data Science, and Machine Learning | 8 |
| Experimental Methods | 5 |
| Qualitative, Descriptive, and Mixed Methods | 3 |

### L2 Frequency

| L2 Sub-method | Papers |
| --- | --- |
| Regression Discontinuity Design (RDD) | 93 |
| Game Theory & Mechanism Design | 10 |
| Macroeconomic & Spatial Equilibrium | 6 |
| Machine Learning Predictors & Regularization | 4 |
| Behavioral & Cognitive Modeling | 4 |
| Survey & Conjoint Experiments | 3 |
| NLP & Text Analysis | 3 |
| Panel Data and Fixed Effects | 3 |
| Other L2 methods | 7 |

### Baseline Comparison

- **L1 agreement with NotebookLM:** 62%
- **L2 agreement with NotebookLM:** 13%

### Known L2 Limitation

RDD is over-assigned (93/145 papers). SPECTER2 produces very similar cosine similarities (~0.89–0.93) across all L2 methods under "Empirical and Econometric Methods", so the z-score threshold amplifies small differences. RDD's semantic description uses generic causal-inference language that slightly outscores the other sub-methods consistently. This is an embedding model granularity limitation, not a pipeline logic issue.

## Notes

- The pdf-to-json conversion is on the `datasource` branch. Some math formulas may not be fully replaced with `[formula]` placeholders.
- NotebookLM assignments (`NotebookLM.csv`) are generated separately and used as a baseline comparison.
