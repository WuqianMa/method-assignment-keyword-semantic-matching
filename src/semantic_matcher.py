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


def _top_k_mean_with_tiers(similarities, tiers, top_k=3, tier2_weight=0.5):
    """Compute tier-weighted top-k mean similarity.

    Tier 1 chunks get full weight (1.0), tier 2 chunks get reduced weight.
    Returns the weighted mean of the top-k highest similarities.
    """
    if not similarities:
        return 0.0

    # Weight each similarity by tier
    weighted = []
    for sim, tier in zip(similarities, tiers):
        weight = 1.0 if tier == 1 else tier2_weight
        weighted.append((sim * weight, weight))

    # Sort by weighted similarity descending
    weighted.sort(key=lambda x: x[0], reverse=True)

    # Take top-k
    top = weighted[:top_k]
    total_weighted_sim = sum(ws for ws, _ in top)
    total_weight = sum(w for _, w in top)

    if total_weight == 0:
        return 0.0

    return total_weighted_sim / total_weight


def match_paper_semantic(chunk_embeddings, chunk_tiers, method_embeddings,
                         top_k=3, tier2_weight=0.5, keyword_hit_counts=None,
                         keyword_bonus_weight=0.1):
    """Compute semantic similarity for one paper against a list of methods.

    Uses top-k mean aggregation with tier weighting instead of max-pool.

    Args:
        chunk_embeddings: list of embedding vectors (one per chunk)
        chunk_tiers: list of tier values (1 or 2) matching chunk_embeddings
        method_embeddings: list of dicts with "name"/"method_name" and "embedding"
        top_k: number of top chunks to average
        tier2_weight: weight for tier 2 chunks
        keyword_hit_counts: optional {method_name: hit_count} for keyword bonus
        keyword_bonus_weight: bonus per keyword hit

    Returns:
        list of (method_name, raw_similarity, combined_score) tuples
    """
    if not chunk_embeddings:
        return []

    results = []
    for method in method_embeddings:
        name = method.get("name") or method.get("method_name")
        method_emb = method["embedding"]

        # Compute similarity of each chunk against this method
        sims = [compute_similarity(ce, method_emb) for ce in chunk_embeddings]

        # Top-k mean with tier weighting
        raw_sim = _top_k_mean_with_tiers(sims, chunk_tiers, top_k, tier2_weight)

        # Add keyword bonus
        bonus = 0.0
        if keyword_hit_counts and name in keyword_hit_counts:
            bonus = keyword_hit_counts[name] * keyword_bonus_weight

        combined = raw_sim + bonus
        results.append((name, round(raw_sim, 4), round(combined, 4)))

    return results


def _apply_zscore_threshold(scores, zscore_threshold=1.0):
    """Filter methods using per-paper z-score threshold.

    Only keeps methods whose combined score is > zscore_threshold standard
    deviations above the paper's mean score.
    """
    if len(scores) < 2:
        return scores

    values = [s[2] for s in scores]  # combined scores
    mean = np.mean(values)
    std = np.std(values)

    if std < 1e-6:
        # All scores are essentially identical — nothing stands out
        return []

    return [s for s in scores if (s[2] - mean) / std > zscore_threshold]


def run_semantic_matching(embedded_chunks_dir, embedded_methods_dir,
                          filter_chunks_dir, keyword_results=None,
                          top_k=3, tier2_weight=0.5,
                          zscore_threshold=1.0, keyword_bonus_weight=0.1):
    """Run semantic matching across all papers with the improved pipeline.

    Args:
        embedded_chunks_dir: directory with embedded chunk JSONs
        embedded_methods_dir: directory with method embedding JSONs
        filter_chunks_dir: directory with filtered chunk JSONs (for tier info)
        keyword_results: optional dict {paper_id: {l1_hit_counts, l2_hit_counts}}
        top_k, tier2_weight, zscore_threshold, keyword_bonus_weight: config params

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
            emb_chunks = json.load(f)

        if not emb_chunks:
            continue

        chunk_embs = [c["embedding"] for c in emb_chunks]

        # Load tier info from filter_chunks
        filter_path = os.path.join(filter_chunks_dir, f"{paper_id}.json")
        if os.path.exists(filter_path):
            with open(filter_path, "r", encoding="utf-8") as f:
                filter_chunks = json.load(f)
            chunk_tiers = [c.get("tier", 1) for c in filter_chunks]
        else:
            chunk_tiers = [1] * len(chunk_embs)

        # Ensure lengths match (safety)
        if len(chunk_tiers) != len(chunk_embs):
            chunk_tiers = [1] * len(chunk_embs)

        # Get keyword hit counts for this paper
        kw_data = keyword_results.get(int(paper_id), {}) if keyword_results else {}
        l1_kw_counts = kw_data.get("l1_hit_counts", {})
        l2_kw_counts = kw_data.get("l2_hit_counts", {})

        # L1 matching
        l1_scores = match_paper_semantic(
            chunk_embs, chunk_tiers, l1_embeddings,
            top_k, tier2_weight, l1_kw_counts, keyword_bonus_weight
        )

        # Apply z-score threshold for L1
        l1_assigned = _apply_zscore_threshold(l1_scores, zscore_threshold)
        matched_l1_names = {s[0] for s in l1_assigned}

        # L2 matching: only consider L2 under assigned L1
        relevant_l2 = [m for m in l2_embeddings if m["level_1_label"] in matched_l1_names]
        l2_scores = match_paper_semantic(
            chunk_embs, chunk_tiers, relevant_l2,
            top_k, tier2_weight, l2_kw_counts, keyword_bonus_weight
        )

        # Apply z-score threshold for L2
        l2_assigned = _apply_zscore_threshold(l2_scores, zscore_threshold)

        # Format output
        l1_names = [s[0] for s in l1_assigned]
        l1_sims = [str(s[1]) for s in l1_assigned]
        l2_names = [s[0] for s in l2_assigned]
        l2_sims = [str(s[1]) for s in l2_assigned]

        results.append({
            "paper_id": int(paper_id),
            "l1_method": "; ".join(l1_names) if l1_names else "",
            "l2_method": "; ".join(l2_names) if l2_names else "",
            "l1_similarity": "; ".join(l1_sims) if l1_sims else "",
            "l2_similarity": "; ".join(l2_sims) if l2_sims else "",
        })

    return results
