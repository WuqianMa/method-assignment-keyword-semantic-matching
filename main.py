import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.keyword_loader import load_l1_methods, load_l2_methods, get_l1_keywords, get_l2_keywords
from src.paper_loader import load_all_papers, load_metadata
from src.keyword_matcher import match_chunks_l1, match_chunks_l2
from src.chunk_filter import save_filtered_chunks
from src.csv_writer import write_keyword_csv, write_semantic_csv
from src.embedder import load_specter2_model, embed_and_save_chunks, embed_and_save_method_descriptions
from src.semantic_matcher import run_semantic_matching
from predefine.config import L1_SIMILARITY_THRESHOLD, L2_SIMILARITY_THRESHOLD, SPECTER2_MODEL_NAME

# Paths
PREDEFINE_DIR = os.path.join(BASE_DIR, "predefine")
DATA_DIR = os.path.join(BASE_DIR, "data")
PAPERS_DIR = os.path.join(DATA_DIR, "papers_json_normalize")
METADATA_PATH = os.path.join(DATA_DIR, "metadata", "meta_data.json")
FILTER_CHUNK_DIR = os.path.join(BASE_DIR, "filter_chunk")
EMBEDDED_CHUNK_DIR = os.path.join(BASE_DIR, "embedded_chunk")
EMBEDDED_METHOD_DIR = os.path.join(BASE_DIR, "embedded_method_description")
ASSIGNMENTS_DIR = os.path.join(BASE_DIR, "assignments")


def run_keyword_pipeline():
    """Stage 1: Keyword matching and chunk filtering."""
    print("=" * 60)
    print("Stage 1: Keyword Matching")
    print("=" * 60)

    # Step 1: Load keyword definitions
    print("\n[1/5] Loading L1/L2 method definitions...")
    l1_methods = load_l1_methods(os.path.join(PREDEFINE_DIR, "l1_methods.json"))
    l2_methods = load_l2_methods(os.path.join(PREDEFINE_DIR, "l2_methods.json"))
    l1_keywords = get_l1_keywords(l1_methods)
    l2_keywords = get_l2_keywords(l2_methods)
    print(f"  Loaded {len(l1_keywords)} L1 categories, {len(l2_keywords)} L2 sub-methods")

    # Step 2: Load all papers
    print("\n[2/5] Loading paper chunks...")
    papers = load_all_papers(PAPERS_DIR)
    print(f"  Loaded {len(papers)} papers")

    # Step 3 & 4: Match and filter
    print("\n[3/5] Running L1 keyword matching (case-sensitive)...")
    keyword_results = []
    matched_count = 0

    for paper_id, chunks in papers.items():
        l1_result = match_chunks_l1(chunks, l1_keywords)
        if not l1_result["matched_chunks"]:
            continue

        matched_count += 1
        # L2 matching on filtered chunks
        l2_methods_found = match_chunks_l2(
            l1_result["matched_chunks"], l2_keywords, l1_result["l1_methods"]
        )

        # Save filtered chunks
        save_filtered_chunks(paper_id, l1_result["matched_chunks"], FILTER_CHUNK_DIR)

        keyword_results.append({
            "paper_id": paper_id,
            "l1_methods": l1_result["l1_methods"],
            "l2_methods": l2_methods_found,
        })

    print(f"  {matched_count}/{len(papers)} papers had keyword matches")
    print(f"  Saved filtered chunks to {FILTER_CHUNK_DIR}")

    # Step 5: Write keyword_match.csv
    print("\n[4/5] Writing keyword_match.csv...")
    csv_path = os.path.join(ASSIGNMENTS_DIR, "keyword_match.csv")
    write_keyword_csv(keyword_results, csv_path)
    print(f"  Written to {csv_path}")

    return l1_methods, l2_methods


def run_embedding_pipeline(l1_methods, l2_methods):
    """Stage 2: SPECTER2 embedding and semantic matching."""
    print("\n" + "=" * 60)
    print("Stage 2: SPECTER2 Embedding & Semantic Matching")
    print("=" * 60)

    # Step 6: Load SPECTER2 model
    print(f"\n[1/4] Loading SPECTER2 model ({SPECTER2_MODEL_NAME})...")
    tokenizer, model = load_specter2_model(SPECTER2_MODEL_NAME)
    print("  Model loaded")

    # Step 7: Embed filtered chunks
    print("\n[2/4] Embedding filtered chunks...")
    embed_and_save_chunks(FILTER_CHUNK_DIR, EMBEDDED_CHUNK_DIR, tokenizer, model)

    # Step 8: Embed method descriptions
    print("\n[3/4] Embedding method descriptions...")
    embed_and_save_method_descriptions(l1_methods, l2_methods, EMBEDDED_METHOD_DIR, tokenizer, model)

    # Step 9 & 10: Semantic matching
    print(f"\n[4/4] Running semantic matching (L1 threshold={L1_SIMILARITY_THRESHOLD}, L2 threshold={L2_SIMILARITY_THRESHOLD})...")
    semantic_results = run_semantic_matching(
        EMBEDDED_CHUNK_DIR, EMBEDDED_METHOD_DIR,
        L1_SIMILARITY_THRESHOLD, L2_SIMILARITY_THRESHOLD,
    )
    csv_path = os.path.join(ASSIGNMENTS_DIR, "semantic_matching.csv")
    write_semantic_csv(semantic_results, csv_path)
    print(f"  Written {len(semantic_results)} results to {csv_path}")


def main():
    print("Method Classification Pipeline")
    print("=" * 60)

    l1_methods, l2_methods = run_keyword_pipeline()
    run_embedding_pipeline(l1_methods, l2_methods)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  - Filtered chunks:    {FILTER_CHUNK_DIR}/")
    print(f"  - Embedded chunks:    {EMBEDDED_CHUNK_DIR}/")
    print(f"  - Method embeddings:  {EMBEDDED_METHOD_DIR}/")
    print(f"  - Keyword CSV:        {os.path.join(ASSIGNMENTS_DIR, 'keyword_match.csv')}")
    print(f"  - Semantic CSV:       {os.path.join(ASSIGNMENTS_DIR, 'semantic_matching.csv')}")
    print(f"\nTo launch the viewer:")
    print(f"  streamlit run app.py")


if __name__ == "__main__":
    main()
