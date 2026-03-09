import re


def _parse_keyword(raw_keyword):
    """Parse a compound keyword into searchable terms.

    Examples:
        "Difference-in-Differences (DiD)" -> ["difference-in-differences", "did"]
        "Two-Stage Least Squares (2SLS)" -> ["two-stage least squares", "2sls"]
        "Heterogeneity-Robust DiD (e.g., Callaway & Sant'Anna, Sun & Abraham)"
            -> ["heterogeneity-robust did"]
        "Fixed Effects" -> ["fixed effects"]

    Returns a list of lowercase search terms.
    """
    raw = raw_keyword.strip()
    terms = []

    # Split on opening parenthesis to get main term and parenthetical
    paren_match = re.match(r'^(.+?)\s*\((.+)\)\s*$', raw)
    if paren_match:
        main_term = paren_match.group(1).strip().lower()
        paren_content = paren_match.group(2).strip()

        terms.append(main_term)

        # Extract abbreviation from parenthetical, skip if starts with "e.g."
        if not paren_content.lower().startswith('e.g.'):
            abbrev = paren_content.strip().lower()
            if abbrev and abbrev != main_term:
                terms.append(abbrev)
    else:
        terms.append(raw.lower())

    return terms


def _build_regex(term):
    """Build a regex pattern for a search term with word-boundary awareness.

    Short abbreviations (<=3 chars) get mandatory word boundaries.
    Longer terms use substring matching.
    """
    escaped = re.escape(term)
    if len(term) <= 3:
        return re.compile(r'\b' + escaped + r'\b', re.IGNORECASE)
    else:
        return re.compile(escaped, re.IGNORECASE)


def match_keywords_in_text(text, keywords):
    """Match a list of raw keywords against text.

    Returns list of matched keyword strings (original form).
    """
    matched = []

    for raw_kw in keywords:
        terms = _parse_keyword(raw_kw)
        for term in terms:
            pattern = _build_regex(term)
            if pattern.search(text):
                matched.append(raw_kw)
                break  # one match per keyword is enough

    return matched


def match_chunks_l1(chunks, l1_keywords):
    """Match L1 keywords against chunks. No longer a hard gate.

    Args:
        chunks: list of chunk dicts (with 'text' and optionally 'tier')
        l1_keywords: {l1_name: [keywords]}

    Returns:
        list of keyword hit dicts: [{chunk_index, keyword, l1_method}, ...]
        set of all matched L1 methods
    """
    all_hits = []
    all_l1_methods = set()

    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        for l1_name, keywords in l1_keywords.items():
            matched = match_keywords_in_text(text, keywords)
            for kw in matched:
                all_hits.append({
                    "chunk_index": idx,
                    "keyword": kw,
                    "l1_method": l1_name,
                })
                all_l1_methods.add(l1_name)

    return all_hits, all_l1_methods


def match_chunks_l2(chunks, l2_keywords, matched_l1_methods):
    """Match L2 keywords, scoped to matched L1 methods.

    Args:
        chunks: list of chunk dicts
        l2_keywords: {l2_method_name: {"level_1_label": ..., "keywords": [...]}}
        matched_l1_methods: set of L1 names that were matched

    Returns:
        list of keyword hit dicts: [{chunk_index, keyword, l2_method, l1_parent}, ...]
        set of all matched L2 methods
    """
    relevant_l2 = {
        name: info
        for name, info in l2_keywords.items()
        if info["level_1_label"] in matched_l1_methods
    }

    all_hits = []
    all_l2_methods = set()

    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        for l2_name, info in relevant_l2.items():
            matched = match_keywords_in_text(text, info["keywords"])
            for kw in matched:
                all_hits.append({
                    "chunk_index": idx,
                    "keyword": kw,
                    "l2_method": l2_name,
                    "l1_parent": info["level_1_label"],
                })
                all_l2_methods.add(l2_name)

    return all_hits, all_l2_methods


def aggregate_keyword_hits(l1_hits, l2_hits, num_chunks):
    """Aggregate keyword hits into per-chunk and per-paper metadata.

    Returns:
        chunk_keyword_meta: list of dicts (one per chunk index) with keyword info
        paper_l1_methods: set of L1 methods
        paper_l2_methods: set of L2 methods
        l1_hit_counts: {l1_name: count}
        l2_hit_counts: {l2_name: count}
    """
    chunk_meta = [{
        "matched_keywords": [],
        "matched_l1_methods": set(),
        "matched_l2_methods": set(),
    } for _ in range(num_chunks)]

    l1_hit_counts = {}
    l2_hit_counts = {}
    paper_l1 = set()
    paper_l2 = set()

    for hit in l1_hits:
        idx = hit["chunk_index"]
        chunk_meta[idx]["matched_keywords"].append(hit["keyword"])
        chunk_meta[idx]["matched_l1_methods"].add(hit["l1_method"])
        paper_l1.add(hit["l1_method"])
        l1_hit_counts[hit["l1_method"]] = l1_hit_counts.get(hit["l1_method"], 0) + 1

    for hit in l2_hits:
        idx = hit["chunk_index"]
        chunk_meta[idx]["matched_keywords"].append(hit["keyword"])
        chunk_meta[idx]["matched_l2_methods"].add(hit["l2_method"])
        paper_l2.add(hit["l2_method"])
        l2_hit_counts[hit["l2_method"]] = l2_hit_counts.get(hit["l2_method"], 0) + 1

    # Convert sets to sorted lists for JSON serialization
    for cm in chunk_meta:
        cm["matched_l1_methods"] = sorted(cm["matched_l1_methods"])
        cm["matched_l2_methods"] = sorted(cm["matched_l2_methods"])

    return chunk_meta, paper_l1, paper_l2, l1_hit_counts, l2_hit_counts
