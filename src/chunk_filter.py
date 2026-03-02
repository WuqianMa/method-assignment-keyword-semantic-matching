import json
import os


def save_filtered_chunks(paper_id, chunks, output_dir):
    """Write filter_chunk/{paper_id}.json with matched chunks."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{paper_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)
