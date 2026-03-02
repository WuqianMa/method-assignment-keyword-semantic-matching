import csv
import os


def write_keyword_csv(results, output_path):
    """
    Write keyword_match.csv.

    Args:
        results: list of dicts with keys: paper_id, l1_methods (set), l2_methods (set)
        output_path: path to output CSV
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["paper_id", "l1_method", "l2_method"])
        for r in sorted(results, key=lambda x: x["paper_id"]):
            l1 = "; ".join(sorted(r["l1_methods"]))
            l2 = "; ".join(sorted(r["l2_methods"]))
            writer.writerow([r["paper_id"], l1, l2])


def write_semantic_csv(results, output_path):
    """
    Write semantic_matching.csv.

    Args:
        results: list of dicts with keys:
            paper_id, l1_method (str), l2_method (str),
            l1_similarity (str), l2_similarity (str)
        output_path: path to output CSV
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["paper_id", "l1_method", "l2_method", "l1_similarity", "l2_similarity"])
        for r in sorted(results, key=lambda x: x["paper_id"]):
            writer.writerow([
                r["paper_id"],
                r["l1_method"],
                r["l2_method"],
                r["l1_similarity"],
                r["l2_similarity"],
            ])
