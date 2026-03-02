import json
import os
import glob
import numpy as np


def compute_similarity(vec_a, vec_b):
    """Cosine similarity between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def match_paper_semantic(chunk_embeddings, l1_embeddings, l2_embeddings, l1_threshold, l2_threshold):
    """
    For one paper's chunk embeddings, compute cosine similarity against method descriptions.

    Args:
        chunk_embeddings: list of embedding vectors (one per chunk)
        l1_embeddings: list of dicts with "name" and "embedding"
        l2_embeddings: list of dicts with "level_1_label", "method_name", "embedding"
        l1_threshold: minimum similarity to assign L1
        l2_threshold: minimum similarity to assign L2

    Returns:
        dict with l1_matches: [{name, similarity}], l2_matches: [{method_name, similarity}]
    """
    # L1 matching: max similarity of any chunk against each L1 description
    l1_matches = []
    for l1 in l1_embeddings:
        max_sim = max(
            compute_similarity(chunk_emb, l1["embedding"])
            for chunk_emb in chunk_embeddings
        )
        if max_sim >= l1_threshold:
            l1_matches.append({"name": l1["name"], "similarity": round(max_sim, 4)})

    matched_l1_names = {m["name"] for m in l1_matches}

    # L2 matching: only consider L2 whose level_1_label is in matched L1
    l2_matches = []
    for l2 in l2_embeddings:
        if l2["level_1_label"] not in matched_l1_names:
            continue
        max_sim = max(
            compute_similarity(chunk_emb, l2["embedding"])
            for chunk_emb in chunk_embeddings
        )
        if max_sim >= l2_threshold:
            l2_matches.append({"method_name": l2["method_name"], "similarity": round(max_sim, 4)})

    return {"l1_matches": l1_matches, "l2_matches": l2_matches}


def run_semantic_matching(embedded_chunks_dir, embedded_methods_dir, l1_threshold, l2_threshold):
    """
    Run semantic matching across all papers.

    Returns:
        list of dicts: paper_id, l1_method, l2_method, l1_similarity, l2_similarity
    """
    # Load method embeddings
    with open(os.path.join(embedded_methods_dir, "l1_embeddings.json"), "r", encoding="utf-8") as f:
        l1_embeddings = json.load(f)
    with open(os.path.join(embedded_methods_dir, "l2_embeddings.json"), "r", encoding="utf-8") as f:
        l2_embeddings = json.load(f)

    results = []
    chunk_files = glob.glob(os.path.join(embedded_chunks_dir, "*.json"))

    for fpath in chunk_files:
        paper_id = os.path.splitext(os.path.basename(fpath))[0]
        with open(fpath, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not chunks:
            continue

        chunk_embs = [c["embedding"] for c in chunks]
        match_result = match_paper_semantic(
            chunk_embs, l1_embeddings, l2_embeddings, l1_threshold, l2_threshold
        )

        l1_names = [m["name"] for m in match_result["l1_matches"]]
        l1_sims = [str(m["similarity"]) for m in match_result["l1_matches"]]
        l2_names = [m["method_name"] for m in match_result["l2_matches"]]
        l2_sims = [str(m["similarity"]) for m in match_result["l2_matches"]]

        results.append({
            "paper_id": int(paper_id),
            "l1_method": "; ".join(l1_names) if l1_names else "",
            "l2_method": "; ".join(l2_names) if l2_names else "",
            "l1_similarity": "; ".join(l1_sims) if l1_sims else "",
            "l2_similarity": "; ".join(l2_sims) if l2_sims else "",
        })

    return results
