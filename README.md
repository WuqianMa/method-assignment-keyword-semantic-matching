# Method Classification Pipeline

Classifies academic papers from the SIOE 2025 conference by research methodology using a two-stage pipeline: **keyword matching** followed by **SPECTER2 semantic similarity**.

Result-preview:
Even though the results are confusing, it's a easy way to inspect the pipeline's result without the ground truth.

https://method-assignment-keyword-semantic-matching-2gxvkogidmtws73fdb.streamlit.app/

Observation:
- 38 out of 150 papers were selected by keywords, I should expand the keyword dictionary.
- NotebookLLm can't be the groundtruth, or maybe i didn't success in filetring all the dialogues that realted to the methods. 
- Methods are quite close to each other, maybe i should improve the semantic meaning of each. 
- I keyword filter first, then do the embedding on the filtered chunks. Maybe embedding all the chunks without selection? 
- The pdf to json is on the datasource branch, I find that i didn't replace all the math formular into [formular], if i am embedding all the chunks these could hurt the similarity check.
- If in the next run i success in expand the vocabulary to filter the related chunks, will it also work on the other years's conferecne? Maybe i should try the 2024 papers too. 
- This is an unmature pipeline everything is in a draft. 
Issue:
Every single one of the 38 papers that passes keyword filtering gets assigned all 5 L1 categories and all 17 L2 methods in semantic matching. The similarity scores cluster in the 0.78–0.94 range, all far above the 0.5 threshold. 
Root causes
1. Max-pooling over many paragraphs guarantees high similarity to everything
In semantic_matcher.py:35-38, each method is scored by max(cosine_similarity) across all of a paper's filtered chunks. A 20-page paper can easily have 50–100+ paragraphs. SPECTER2 embeddings of academic economics text already live in a high-similarity neighborhood (baseline ~0.80+). Taking the max over dozens of chunks ensures at least one chunk will score high against any method description. More chunks = higher max = everything matches.
This is the core "naive indexing" problem. The pipeline treats paragraphs as independent probes and takes the best one — which statistically guarantees a match.
2. The 0.5 threshold is meaningless for SPECTER2
In predefine/config.py:1-2, both thresholds are 0.5. But SPECTER2 embeddings of academic text never produce cosine similarities near 0.5 — the floor for any two economics paragraphs is already ~0.75-0.80. The threshold would need to be somewhere in the 0.90–0.95 range to be discriminative, and even then max-pooling makes it fragile.
3. Keyword filtering is simultaneously too aggressive (drops 75% of papers) and too loose (lets noise through)
In keyword_matcher.py:4-6:
Case-sensitive exact substring match — keyword in text has no word boundary checking. "Staggered" will match inside "unStaggered" or any compound. Conversely, "staggered" (lowercase) in body text won't match the keyword "Staggered".
Compound keywords never match — Keywords like "Heterogeneity-Robust DiD (e.g., Callaway & Sant'Anna, Sun & Abraham)" are 70+ character strings including parenthetical descriptions. No paper text will contain this exact substring. These are dead keywords.
No stemming or fuzzy matching — "Instrumental Variable" won't match the keyword "Instrumental Variables (IV)". "difference-in-differences" won't match "Difference-in-Differences (DiD)".
Result: 112/150 papers are silently discarded, likely including many that genuinely use methods from the taxonomy.
4. No structural awareness — all paragraphs are treated equally
The data includes section_name (e.g., "4. Empirical strategy and results", "Introduction", "References") but the pipeline ignores it entirely. A paragraph from the references section, a literature review ("Prior work used DiD..."), or a figure caption gets the same weight as the actual methodology section. This is a major source of false positives — a paper that cites a method is not the same as a paper that uses it.
5. No filtering of junk chunks
Looking at sample papers, many chunks are:
Author affiliations: "Robert Akerlof, UNSW Business School..."
Figure captions: "(a) Most people do not see all five horses immediately."
JEL codes: "JEL Classification: D01, D80, D90."
Single keywords lines: "Keywords: Cognition, reasoning, perception, narratives."
These all get embedded and participate in max-pooling, adding noise.
6. The L1→L2 scoping provides no constraint
The design in semantic_matcher.py:46-47 scopes L2 matches to only L2 methods whose parent L1 matched. But since every L1 always matches (see point 1), this scoping does nothing.
What would actually improve this
Here are concrete architectural changes, roughly ordered by impact:
Replace max-pooling with mean or weighted aggregation — Average the cosine similarity across chunks rather than taking the max. Or use top-k averaging (e.g., top 3 chunks). This penalizes methods that only weakly appear once.
Section-aware filtering — Only embed paragraphs from methodology-relevant sections (filter on section_name containing patterns like "method", "empirical", "strategy", "data", "estimation", "identification", "experiment", "model"). Exclude references, acknowledgments, appendix, and abstract sections from the matching pool.
Minimum chunk length threshold — Skip chunks shorter than, say, 50 characters. This removes figure captions, author lines, and JEL codes.
Fix keyword matching — Normalize both keywords and text to lowercase. Strip parenthetical annotations from keywords before matching (e.g., "Instrumental Variables (IV)" → search for both "instrumental variables" and "iv" independently). Add word-boundary awareness.
Raise or dynamically calibrate the threshold — Either set it empirically (look at the similarity distribution and pick a threshold that separates known positives from negatives, likely 0.88–0.92), or use a relative approach (rank methods by similarity and take only those significantly above the mean).
Embed at the section level, not paragraph level — Concatenate paragraphs within the same section and embed the section as a whole. This gives SPECTER2 more context to work with and naturally reduces the max-pooling problem by having fewer, more meaningful vectors.
Embed all 150 papers, not just keyword-filtered ones — The keyword gate discards too many papers. If the semantic stage worked properly, it could classify papers independently without needing the keyword pre-filter.
The fundamental issue is that this pipeline was designed as "keyword filter first, then refine with embeddings," but the refinement step has no teeth. It needs either a much stronger aggregation strategy, section-level awareness, or both.


## Project Structure

```
method/
├── main.py                            # Pipeline entry point
├── app.py                             # Streamlit viewer (run separately)
├── requirements.txt                   # Streamlit dependency for app.py
│
├── predefine/                         # Method taxonomy & config
│   ├── l1_methods.json                # 5 L1 categories with keywords
│   ├── l2_methods.json                # 17 L2 sub-methods (linked to L1 via level_1_label)
│   ├── json_describe.md               # Schema docs for the above
│   └── config.py                      # Thresholds & model name
│
├── data/                              # Input data (read-only)
│   ├── metadata/meta_data.json        # 150 papers: paper_id, title, author, etc.
│   ├── papers_json_normalize/*.json   # 150 files, paragraph-level chunks per paper
│   └── data_describe.md
│
├── src/                               # Pipeline modules
│   ├── keyword_loader.py              # Load L1/L2 method definitions
│   ├── paper_loader.py                # Load paper JSONs and metadata
│   ├── keyword_matcher.py             # Case-sensitive keyword matching
│   ├── chunk_filter.py                # Save filtered chunks to disk
│   ├── csv_writer.py                  # Write assignment CSVs
│   ├── embedder.py                    # SPECTER2 embedding
│   └── semantic_matcher.py            # Cosine similarity matching
│
├── filter_chunk/                      # Output: keyword-filtered chunks
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

The pipeline runs in two stages with a strict hierarchical filtering order: **all chunks -> L1 keyword filter -> filtered chunks -> L2 keyword match (scoped to matched L1) -> SPECTER2 embed filtered chunks only -> L1 semantic match -> L2 semantic match (scoped to matched L1)**.

### Stage 1: Keyword Matching

```
150 papers (all paragraph chunks)
        │
        ▼
   ┌─────────────────────────────────┐
   │  L1 Keyword Match               │
   │  Case-sensitive substring scan  │
   │  against l1_methods.json        │
   │  keywords (5 categories,        │
   │  ~135 keywords total)           │
   └─────────────┬───────────────────┘
                 │
        Only chunks containing        Papers with zero
        at least one L1 keyword        L1 matches are
        are kept                       discarded entirely
                 │
                 ▼
   ┌─────────────────────────────────┐
   │  Save to filter_chunk/          │
   │  {paper_id}.json                │
   │  (same schema + matched_keywords│
   │   matched_l1_methods fields)    │
   └─────────────┬───────────────────┘
                 │
                 ▼
   ┌─────────────────────────────────┐
   │  L2 Keyword Match               │
   │  Only L2 methods whose          │
   │  level_1_label is in the        │
   │  paper's matched L1 set         │
   │  are considered.                │
   │  Scans filtered chunks only.    │
   └─────────────┬───────────────────┘
                 │
                 ▼
   ┌─────────────────────────────────┐
   │  Write keyword_match.csv        │
   │  Columns: paper_id, l1_method,  │
   │  l2_method                      │
   │  Multiple methods joined by "; "│
   └─────────────────────────────────┘
```

**Key rules:**
- All keyword matching is **case-sensitive** (these are formal academic terms like "DiD", "2SLS", "RDD")
- A chunk must contain at least one L1 keyword to be kept
- L2 matching is **scoped**: only L2 methods belonging to a matched L1 category are checked
- A paper can match multiple L1 and L2 methods

### Stage 2: SPECTER2 Semantic Matching

This stage operates **only on filtered chunks** from Stage 1.

```
   filter_chunk/{paper_id}.json          predefine/l1_methods.json
   (keyword-filtered chunks)             predefine/l2_methods.json
                │                                    │
                ▼                                    ▼
   ┌────────────────────────┐       ┌────────────────────────────┐
   │  SPECTER2 Embed Chunks │       │  SPECTER2 Embed            │
   │  Each chunk's text     │       │  semantic_meaning field     │
   │  → 768-dim vector      │       │  of each L1 and L2 method  │
   │  Saved to              │       │  Saved to                  │
   │  embedded_chunk/       │       │  embedded_method_description│
   └──────────┬─────────────┘       └──────────┬─────────────────┘
              │                                │
              └───────────┬────────────────────┘
                          ▼
   ┌──────────────────────────────────────────┐
   │  L1 Cosine Similarity                    │
   │  For each paper:                         │
   │    max(cosine(chunk_i, L1_j))            │
   │    across all chunks, for each L1        │
   │  Assign L1 if max_sim >= L1_threshold    │
   └────────────────┬─────────────────────────┘
                    │
                    ▼
   ┌──────────────────────────────────────────┐
   │  L2 Cosine Similarity (scoped to L1)     │
   │  Only compare against L2 methods whose   │
   │  level_1_label matches an assigned L1     │
   │  Assign L2 if max_sim >= L2_threshold     │
   └────────────────┬─────────────────────────┘
                    │
                    ▼
   ┌──────────────────────────────────────────┐
   │  Write semantic_matching.csv             │
   │  Columns: paper_id, l1_method, l2_method,│
   │  l1_similarity, l2_similarity            │
   └──────────────────────────────────────────┘
```

**Key rules:**
- SPECTER2 base model (`allenai/specter2_base`) is used without task-specific adapters
- Each chunk is embedded individually (chunks are within the 512-token window)
- Similarity is the **max** cosine similarity of any chunk against each method description
- L2 semantic matching is **scoped to L1**: only L2 methods under an assigned L1 are compared
- Thresholds are configured in `predefine/config.py` (default: 0.5 for CSV generation)

## Method Taxonomy

### L1 Categories (5), L2 Sub-methods (17)

| # | Category | L2 Sub-methods |
| --- | --- | --- |
| 1 | Empirical and Econometric Methods | [Panel Data and Fixed Effects, Difference-in-Differences (DiD) & Event Studies, Instrumental Variables (IV) & Shift-Share Designs, Regression Discontinuity Design (RDD), Structural & Advanced Econometrics] |
| 2 | Computational, Data Science, and Machine Learning | [Natural Language Processing (NLP) & Text Analysis, Machine Learning Predictors & Regularization, Network & Spatial Analysis] |
| 3 | Theoretical and Formal Modeling | [Game Theory & Mechanism Design, Behavioral & Cognitive Modeling, Macroeconomic & Spatial Equilibrium] |
| 4 | Experimental Methods | [Field Experiments / RCTs, Laboratory & Online Experiments, Survey & Conjoint Experiments] |
| 5 | Qualitative, Descriptive, and Mixed Methods | [Archival Research & Case Studies, Interviews, Literature Reviews & Aggregation] |

---
## Results

With case-sensitive keyword matching on 150 papers:
- **38 papers** had at least one L1 keyword match (25.3%)
- **112 papers** had no keyword matches and were excluded from further processing
- Filtered chunks, embeddings, and semantic similarity scores were computed for the 38 matched papers

Output files:
- `assignments/keyword_match.csv` — 38 rows with L1/L2 keyword assignments
- `assignments/semantic_matching.csv` — 38 rows with L1/L2 semantic assignments and similarity scores
- `assignments/NotebookLM.csv` — 149 rows (baseline, generated separately via NotebookLM RAG)

## Usage

### Run the pipeline

```bash
cd method
python main.py
```

Requires: `transformers`, `torch`, `numpy`

### Launch the viewer

```bash
cd method
streamlit run app.py
```

Requires: `streamlit` (see `requirements.txt`)

The viewer provides:
- **Prev/Next navigation** between papers with filtered chunks
- **Filtered chunk display** with matched keywords highlighted in orange
- **Three assignment columns**: NotebookLM (baseline), Keyword Match, Semantic Match
- **L1/L2 threshold sliders** in the sidebar that dynamically show/hide semantic labels
- L2 methods grouped under their parent L1 in both keyword and semantic displays

### Configuration

Edit `predefine/config.py` to change:
- `L1_SIMILARITY_THRESHOLD` — minimum cosine similarity to assign an L1 method (default: 0.5)
- `L2_SIMILARITY_THRESHOLD` — minimum cosine similarity to assign an L2 method (default: 0.5)
- `SPECTER2_MODEL_NAME` — HuggingFace model identifier (default: `allenai/specter2_base`)

The Streamlit app's sidebar sliders override these thresholds for display purposes without re-running the pipeline.
