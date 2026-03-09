import re

# Section name patterns for tiering (matched case-insensitively against section_name)
TIER1_PATTERNS = [
    r'empirical', r'strategy', r'method', r'model', r'identification',
    r'estimation', r'experiment', r'design', r'analysis', r'results?',
    r'data\b', r'robustness', r'mechanism', r'framework', r'approach',
    r'regression', r'instrumental', r'causal', r'treatment', r'variable',
    r'specification', r'heterogeneity', r'sample', r'discontinuity',
    r'difference', r'panel', r'fixed.effect', r'event.stud',
]

TIER2_PATTERNS = [
    r'introduction', r'background', r'discussion', r'conclusion',
    r'literature', r'related', r'survey', r'review', r'motivation',
    r'overview', r'setting', r'context',
]

TIER3_PATTERNS = [
    r'^preamble$', r'appendix', r'^references?$', r'proofs?',
    r'acknowledg', r'^abstract:?$', r'^unknown$', r'bibliography',
    r'^figures?$', r'^tables?$', r'^ABSTRACT$',
]


def classify_section_tier(section_name):
    """Classify a section name into tier 1 (methodology), 2 (contextual), or 3 (excluded).

    Unrecognized section names default to tier 1 (assume relevant since many
    papers have specific section titles).
    """
    if not section_name:
        return 3

    name_lower = section_name.lower().strip()

    # Strip leading section numbers like "4.1.", "3 . 2", "2."
    name_clean = re.sub(r'^[\d]+[\s.]*[\d]*[\s.]*', '', name_lower).strip()

    # Check tier 3 (excluded) first
    for pattern in TIER3_PATTERNS:
        if re.search(pattern, name_clean, re.IGNORECASE):
            return 3

    # Check tier 2 (contextual)
    for pattern in TIER2_PATTERNS:
        if re.search(pattern, name_clean, re.IGNORECASE):
            return 2

    # Check tier 1 (methodology-relevant)
    for pattern in TIER1_PATTERNS:
        if re.search(pattern, name_clean, re.IGNORECASE):
            return 1

    # Default: unrecognized section names are tier 1 (likely paper-specific titles)
    return 1


def filter_paper_chunks(chunks, min_length=100):
    """Filter a single paper's chunks based on section tier and minimum length.

    Returns list of chunks that pass filtering, each annotated with a 'tier' field.
    Tier 3 chunks are excluded entirely.
    """
    filtered = []
    for chunk in chunks:
        text = chunk.get("text", "")

        # Minimum length filter
        if len(text) < min_length:
            continue

        # Section tier classification
        section_name = chunk.get("section_name", "")
        tier = classify_section_tier(section_name)

        # Exclude tier 3
        if tier == 3:
            continue

        # Keep tier 1 and 2 with annotation
        filtered_chunk = dict(chunk)
        filtered_chunk["tier"] = tier
        filtered.append(filtered_chunk)

    return filtered
