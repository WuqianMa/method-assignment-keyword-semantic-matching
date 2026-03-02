import json
import os
import glob


def load_paper_chunks(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_papers(folder_path):
    """Load all paper JSONs, keyed by paper_id."""
    papers = {}
    for fpath in glob.glob(os.path.join(folder_path, "*.json")):
        chunks = load_paper_chunks(fpath)
        if chunks:
            paper_id = chunks[0]["paper_id"]
            papers[paper_id] = chunks
    return papers


def load_metadata(meta_path):
    """Load metadata keyed by paper_id."""
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["paper_id"]: item for item in data}
