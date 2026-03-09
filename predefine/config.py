SPECTER2_MODEL_NAME = "allenai/specter2_base"

# Paragraph filtering (method-agnostic)
MIN_CHUNK_LENGTH = 100

# Semantic matching
TOP_K = 3
TIER2_WEIGHT = 0.5
ZSCORE_THRESHOLD = 1.0   # methods must be >1 std above paper mean

# Keyword matching
KEYWORD_BONUS_WEIGHT = 0.1  # bonus added to semantic score per keyword hit

# Legacy thresholds (used by app.py slider defaults)
L1_SIMILARITY_THRESHOLD = 0.5
L2_SIMILARITY_THRESHOLD = 0.5
