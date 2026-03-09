import json
import os
import glob
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch


def load_specter2_model(model_name="allenai/specter2_base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def embed_texts(texts, tokenizer, model, batch_size=16):
    """Batch-embed a list of text strings, return numpy array of vectors."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs)
        # Use [CLS] token embedding
        embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(embeddings.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


def embed_and_save_chunks(filtered_chunks_dir, output_dir, tokenizer, model, min_length=100):
    """
    For each filter_chunk/{paper_id}.json, embed each chunk's text field.
    Save to embedded_chunk/{paper_id}.json with text replaced by embedding.
    Operates on ALL filtered papers (method-agnostic), not just keyword-matched ones.
    Skips chunks shorter than min_length as a safety net.
    """
    os.makedirs(output_dir, exist_ok=True)
    chunk_files = glob.glob(os.path.join(filtered_chunks_dir, "*.json"))

    for fpath in chunk_files:
        paper_id = os.path.splitext(os.path.basename(fpath))[0]
        with open(fpath, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not chunks:
            continue

        # Safety net: skip very short chunks
        valid_chunks = [c for c in chunks if len(c.get("text", "")) >= min_length]
        if not valid_chunks:
            continue

        texts = [c["text"] for c in valid_chunks]
        embeddings = embed_texts(texts, tokenizer, model)

        embedded_chunks = []
        for chunk, emb in zip(valid_chunks, embeddings):
            ec = {k: v for k, v in chunk.items() if k != "text"}
            ec["embedding"] = emb.tolist()
            embedded_chunks.append(ec)

        out_path = os.path.join(output_dir, f"{paper_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(embedded_chunks, f, ensure_ascii=False)

    print(f"  Embedded chunks for {len(chunk_files)} papers -> {output_dir}")


def embed_and_save_method_descriptions(l1_methods, l2_methods, output_dir, tokenizer, model):
    """
    Embed each method's semantic_meaning.
    Save to embedded_method_description/l1_embeddings.json and l2_embeddings.json.
    """
    os.makedirs(output_dir, exist_ok=True)

    # L1
    l1_texts = [m["semantic_meaning"] for m in l1_methods]
    l1_embeddings = embed_texts(l1_texts, tokenizer, model)
    l1_out = []
    for m, emb in zip(l1_methods, l1_embeddings):
        l1_out.append({
            "name": m["name"],
            "semantic_meaning": m["semantic_meaning"],
            "embedding": emb.tolist(),
        })
    with open(os.path.join(output_dir, "l1_embeddings.json"), "w", encoding="utf-8") as f:
        json.dump(l1_out, f, indent=4, ensure_ascii=False)

    # L2
    l2_texts = [m["semantic_meaning"] for m in l2_methods]
    l2_embeddings = embed_texts(l2_texts, tokenizer, model)
    l2_out = []
    for m, emb in zip(l2_methods, l2_embeddings):
        l2_out.append({
            "level_1_label": m["level_1_label"],
            "method_name": m["method_name"],
            "semantic_meaning": m["semantic_meaning"],
            "embedding": emb.tolist(),
        })
    with open(os.path.join(output_dir, "l2_embeddings.json"), "w", encoding="utf-8") as f:
        json.dump(l2_out, f, indent=4, ensure_ascii=False)

    print(f"  Embedded {len(l1_methods)} L1 and {len(l2_methods)} L2 method descriptions -> {output_dir}")
