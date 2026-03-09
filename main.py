import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.keyword_loader import load_l1_methods, load_l2_methods, get_l1_keywords, get_l2_keywords
from src.paper_loader import load_all_papers
from src.paragraph_filter import filter_paper_chunks
from src.keyword_matcher import match_chunks_l1, match_chunks_l2, aggregate_keyword_hits
from src.chunk_filter import save_filtered_chunks
from src.csv_writer import write_keyword_csv, write_semantic_csv
from src.embedder import load_specter2_model, embed_and_save_chunks, embed_and_save_method_descriptions
from src.semantic_matcher import run_semantic_matching
from predefine.config import (
    SPECTER2_MODEL_NAME, MIN_CHUNK_LENGTH,
    TOP_K, TIER2_WEIGHT, ZSCORE_THRESHOLD, KEYWORD_BONUS_WEIGHT,
)

# Paths
PREDEFINE_DIR = os.path.join(BASE_DIR, "predefine")
DATA_DIR = os.path.join(BASE_DIR, "data")
PAPERS_DIR = os.path.join(DATA_DIR, "papers_json_normalize")
FILTER_CHUNK_DIR = os.path.join(BASE_DIR, "filter_chunk")
EMBEDDED_CHUNK_DIR = os.path.join(BASE_DIR, "embedded_chunk")
EMBEDDED_METHOD_DIR = os.path.join(BASE_DIR, "embedded_method_description")
ASSIGNMENTS_DIR = os.path.join(BASE_DIR, "assignments")


def run_stage1_paragraph_filtering(papers):
    """Stage 1: Method-agnostic paragraph filtering."""
    print("=" * 60)
    print("Stage 1: Paragraph Filtering (method-agnostic)")
    print("=" * 60)

    total_chunks = 0
    total_filtered = 0

    for paper_id, chunks in papers.items():
        filtered = filter_paper_chunks(chunks, min_length=MIN_CHUNK_LENGTH)
        if filtered:
            save_filtered_chunks(paper_id, filtered, FILTER_CHUNK_DIR)
            total_filtered += 1
            total_chunks += len(filtered)

    print(f"  {total_filtered}/{len(papers)} papers have chunks after filtering")
    print(f"  {total_chunks} total chunks saved to {FILTER_CHUNK_DIR}/")
    return total_filtered


def run_stage2_keyword_matching(papers, l1_methods, l2_methods):
    """Stage 2: Keyword matching as a soft signal."""
    print("\n" + "=" * 60)
    print("Stage 2: Keyword Matching (soft signal)")
    print("=" * 60)

    l1_keywords = get_l1_keywords(l1_methods)
    l2_keywords = get_l2_keywords(l2_methods)
    print(f"  {len(l1_keywords)} L1 categories, {len(l2_keywords)} L2 sub-methods")

    keyword_csv_results = []
    keyword_results = {}  # {paper_id: {l1_hit_counts, l2_hit_counts}}
    papers_with_hits = 0

    for paper_id, chunks in papers.items():
        # Load filtered chunks (with tier info)
        import json
        filter_path = os.path.join(FILTER_CHUNK_DIR, f"{paper_id}.json")
        if not os.path.exists(filter_path):
            continue

        with open(filter_path, "r", encoding="utf-8") as f:
            filtered_chunks = json.load(f)

        if not filtered_chunks:
            continue

        # L1 keyword matching
        l1_hits, l1_methods_found = match_chunks_l1(filtered_chunks, l1_keywords)

        # L2 keyword matching (scoped to matched L1)
        l2_hits, l2_methods_found = match_chunks_l2(
            filtered_chunks, l2_keywords, l1_methods_found
        )

        # Aggregate
        chunk_meta, paper_l1, paper_l2, l1_counts, l2_counts = aggregate_keyword_hits(
            l1_hits, l2_hits, len(filtered_chunks)
        )

        # Attach keyword metadata to filtered chunks and re-save
        for i, cm in enumerate(chunk_meta):
            filtered_chunks[i]["matched_keywords"] = cm["matched_keywords"]
            filtered_chunks[i]["matched_l1_methods"] = cm["matched_l1_methods"]
            filtered_chunks[i]["matched_l2_methods"] = cm["matched_l2_methods"]

        save_filtered_chunks(paper_id, filtered_chunks, FILTER_CHUNK_DIR)

        if paper_l1:
            papers_with_hits += 1

        keyword_csv_results.append({
            "paper_id": paper_id,
            "l1_methods": paper_l1,
            "l2_methods": paper_l2,
        })

        keyword_results[paper_id] = {
            "l1_hit_counts": l1_counts,
            "l2_hit_counts": l2_counts,
        }

    print(f"  {papers_with_hits}/{len(papers)} papers had keyword matches")

    # Write keyword_match.csv
    csv_path = os.path.join(ASSIGNMENTS_DIR, "keyword_match.csv")
    write_keyword_csv(keyword_csv_results, csv_path)
    print(f"  Written to {csv_path}")

    return keyword_results


def run_stage3_embedding(l1_methods, l2_methods):
    """Stage 3: SPECTER2 embedding of all filtered chunks."""
    print("\n" + "=" * 60)
    print("Stage 3: SPECTER2 Embedding")
    print("=" * 60)

    print(f"\n  Loading SPECTER2 model ({SPECTER2_MODEL_NAME})...")
    tokenizer, model = load_specter2_model(SPECTER2_MODEL_NAME)
    print("  Model loaded")

    print("\n  Embedding all filtered chunks (all papers)...")
    embed_and_save_chunks(FILTER_CHUNK_DIR, EMBEDDED_CHUNK_DIR, tokenizer, model,
                          min_length=MIN_CHUNK_LENGTH)

    print("\n  Embedding method descriptions...")
    embed_and_save_method_descriptions(l1_methods, l2_methods, EMBEDDED_METHOD_DIR, tokenizer, model)


def run_stage4_semantic_matching(keyword_results):
    """Stage 4: Semantic + combined matching."""
    print("\n" + "=" * 60)
    print("Stage 4: Semantic + Combined Matching")
    print("=" * 60)

    print(f"  top_k={TOP_K}, tier2_weight={TIER2_WEIGHT}, "
          f"zscore_threshold={ZSCORE_THRESHOLD}, keyword_bonus={KEYWORD_BONUS_WEIGHT}")

    semantic_results = run_semantic_matching(
        EMBEDDED_CHUNK_DIR, EMBEDDED_METHOD_DIR, FILTER_CHUNK_DIR,
        keyword_results=keyword_results,
        top_k=TOP_K, tier2_weight=TIER2_WEIGHT,
        zscore_threshold=ZSCORE_THRESHOLD, keyword_bonus_weight=KEYWORD_BONUS_WEIGHT,
    )

    csv_path = os.path.join(ASSIGNMENTS_DIR, "semantic_matching.csv")
    write_semantic_csv(semantic_results, csv_path)
    print(f"  Written {len(semantic_results)} results to {csv_path}")


def main():
    print("Method Classification Pipeline (4-Stage)")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    l1_methods = load_l1_methods(os.path.join(PREDEFINE_DIR, "l1_methods.json"))
    l2_methods = load_l2_methods(os.path.join(PREDEFINE_DIR, "l2_methods.json"))
    papers = load_all_papers(PAPERS_DIR)
    print(f"  Loaded {len(papers)} papers, {len(l1_methods)} L1, {len(l2_methods)} L2 methods\n")

    # Stage 1: Paragraph Filtering (method-agnostic)
    run_stage1_paragraph_filtering(papers)

    # Stage 2: Keyword Matching (soft signal)
    keyword_results = run_stage2_keyword_matching(papers, l1_methods, l2_methods)

    # Stage 3: SPECTER2 Embedding
    run_stage3_embedding(l1_methods, l2_methods)

    # Stage 4: Semantic + Combined Matching
    run_stage4_semantic_matching(keyword_results)

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
