import copy


def find_keyword_in_text(text, keyword):
    """Case-sensitive substring check."""
    return keyword in text


def match_chunks_l1(chunks, l1_keywords):
    """
    For a single paper's chunks, find which chunks contain any L1 keyword.
    Case-sensitive substring search.

    Args:
        chunks: list of chunk dicts from a paper JSON
        l1_keywords: {l1_name: [keywords]} from keyword_loader.get_l1_keywords()

    Returns:
        {"matched_chunks": [...], "l1_methods": set()}
        Each matched chunk gets: matched_keywords, matched_l1_methods, matched_l2_methods (empty initially)
    """
    matched_chunks = []
    all_l1_methods = set()

    for chunk in chunks:
        text = chunk["text"]
        chunk_keywords = []
        chunk_l1 = set()

        for l1_name, keywords in l1_keywords.items():
            for kw in keywords:
                if find_keyword_in_text(text, kw):
                    chunk_keywords.append(kw)
                    chunk_l1.add(l1_name)

        if chunk_keywords:
            matched = copy.deepcopy(chunk)
            matched["matched_keywords"] = chunk_keywords
            matched["matched_l1_methods"] = sorted(chunk_l1)
            matched["matched_l2_methods"] = []
            matched_chunks.append(matched)
            all_l1_methods.update(chunk_l1)

    return {"matched_chunks": matched_chunks, "l1_methods": all_l1_methods}


def match_chunks_l2(filtered_chunks, l2_keywords, matched_l1_methods):
    """
    Within already-filtered chunks, match L2 keywords.
    L2 matching only considers L2 methods whose level_1_label is in matched_l1_methods.

    Args:
        filtered_chunks: list of chunk dicts (already L1-matched, with matched_* fields)
        l2_keywords: {l2_method_name: {"level_1_label": ..., "keywords": [...]}}
        matched_l1_methods: set of L1 method names matched for this paper

    Returns:
        set of matched L2 method names. Also annotates chunks in-place with matched_l2_methods.
    """
    relevant_l2 = {
        name: info
        for name, info in l2_keywords.items()
        if info["level_1_label"] in matched_l1_methods
    }

    all_l2_methods = set()

    for chunk in filtered_chunks:
        text = chunk["text"]
        chunk_l2 = set()

        for l2_name, info in relevant_l2.items():
            for kw in info["keywords"]:
                if find_keyword_in_text(text, kw):
                    chunk_l2.add(l2_name)
                    break

        if chunk_l2:
            chunk["matched_l2_methods"] = sorted(chunk_l2)
            all_l2_methods.update(chunk_l2)

    return all_l2_methods
