#!/usr/bin/env python3
"""
Text cleaning for extracted academic paper content.
Removes math formulas, tables, in-text citations, and other noise.
Designed to produce clean text suitable for embedding / NLP.
"""

import re
import unicodedata


# =============================================================================
# PRE-NORMALIZATION
# =============================================================================

# Standalone diacritical marks → corresponding Unicode combining marks.
# PDF extractors often emit these standalone characters before the letter
# they should modify (e.g. ¸c instead of ç).
_DIACRITICS_MAP = {
    '\u00B8': '\u0327',  # ¸ CEDILLA         → COMBINING CEDILLA
    '\u02DC': '\u0303',  # ˜ SMALL TILDE     → COMBINING TILDE
    '\u00B4': '\u0301',  # ´ ACUTE ACCENT    → COMBINING ACUTE ACCENT
    '\u02C6': '\u0302',  # ˆ MODIFIER CIRC.  → COMBINING CIRCUMFLEX
    '\u00A8': '\u0308',  # ¨ DIAERESIS       → COMBINING DIAERESIS
    '\u02C7': '\u030C',  # ˇ CARON           → COMBINING CARON
    '\u02D8': '\u0306',  # ˘ BREVE           → COMBINING BREVE
    '\u02DA': '\u030A',  # ˚ RING ABOVE      → COMBINING RING ABOVE
    '\u00AF': '\u0304',  # ¯ MACRON          → COMBINING MACRON
    '\u02BB': '\u0300',  # ʻ MOD. TURNED COMMA → COMBINING GRAVE (common PDF artifact)
}


def _pre_normalize(text: str) -> str:
    """Fix PDF extraction artifacts before any pattern matching.

    Must run FIRST so that dehyphenation and control-char fixes are in place
    before math/citation regexes are applied.
    """
    # PDF bracket artifacts (common in math-heavy papers): U+0002 → '[', U+0003 → ']'
    text = text.replace('\u0002', '[').replace('\u0003', ']')
    # Other non-printable control characters → space
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)
    # Fix broken diacritics from PDF extraction.
    # PDF extractors often produce standalone diacritical marks BEFORE the
    # letter they modify (e.g. "Assun¸c˜ao" instead of "Assunção").
    # Step A: fix dotless-i artifact FIRST so it becomes ASCII (ı → i)
    text = text.replace('\u0131', 'i')
    # Step B: replace standalone-mark + letter → letter + combining-mark
    for _standalone, _combining in _DIACRITICS_MAP.items():
        text = re.sub(
            r'\s*' + re.escape(_standalone) + r'\s*([a-zA-Z])',
            r'\1' + _combining, text
        )
    # Step C: also fix combining marks (U+0300–U+036F) separated by spaces
    text = re.sub(r'\s([\u0300-\u036f])(\w)', r'\2\1', text)
    # NFKC normalize early — composes diacritics, converts fullwidth chars
    # to ASCII, resolves ligatures (fi→fi), etc.  Ensures citation/math
    # regexes see clean text.
    text = unicodedata.normalize('NFKC', text)
    # Dehyphenate line-break hyphenation: "institu- tion" → "institution"
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    # Normalise line endings
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# =============================================================================
# MATH / FORMULA PATTERNS
# =============================================================================

# Greek letters and math symbols (Unicode)
GREEK_MATH_SYMBOLS = (
    r'[αβγδεζηθικλμνξπρστυφχψω'
    r'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ'
    r'∗∂∇∞∑∏∫∈∉⊂⊃⊆⊇∧∨¬⇒⇔→←↔'
    r'≤≥≈≡≠∼∝±×÷√∝∅∃∀]'
)

MATH_PATTERNS = [
    # LaTeX inline math: $...$
    re.compile(r'\$[^$]+\$'),
    # LaTeX display math: \(...\) and \[...\]
    re.compile(r'\\\(.*?\\\)', re.DOTALL),
    re.compile(r'\\\[.*?\\\]', re.DOTALL),
    # Backslash-prefixed expressions (PDF LaTeX artefacts): "\\ Var [...]"
    re.compile(r'\\{1,2}\s+\w[\w\s∗²³\[\]{}()=<>+\-*/.,|∈∉]{5,}'),
    # Common LaTeX commands embedded in text
    re.compile(r'\\(?:frac|sqrt|sum|prod|int|lim|log|ln|exp|sin|cos|tan|'
               r'alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|omega|'
               r'partial|nabla|infty|approx|equiv|leq|geq|neq|subset|supset|'
               r'cap|cup|forall|exists|mathbb|mathcal|mathrm|textit|textbf|'
               r'hat|bar|tilde|vec|dot|ddot|overline|underline)'
               r'(?:\{[^}]*\}|\b)'),
    # Sequences of math-like symbols (3+ non-word chars in a row)
    re.compile(r'[^\w\s,.;:!?\'"()\[\]\-/]{3,}'),
]

# Patterns for "broken" math — symbols extracted from PDF without LaTeX markup
BROKEN_MATH_PATTERNS = [
    # Inline broken formulas: two Greek/math symbols with tokens between them
    re.compile(
        r'(?:' + GREEK_MATH_SYMBOLS + r')'
        r'[\w\s∗∂∇∞≤≥≈∼∝±×÷=<>+\-*/^_{}()\[\]|,.:!²³⁴⁵⁶⁷⁸⁹⁰¹]*'
        r'(?:' + GREEK_MATH_SYMBOLS + r')'
        r'[\w\s∗∂∇∞≤≥≈∼∝±×÷=<>+\-*/^_{}()\[\]|,.:²³⁴⁵⁶⁷⁸⁹⁰¹]*'
    ),
    # Equation with number at end: "... = ... . (2)" or "... , (3)"
    re.compile(r'[=<>≤≥≈∼∝][^.]*\.\s*\(\d+\)'),
    # Standalone equation labels at end of text: "(1)" "(2)" etc.
    re.compile(r'\s*\(\d{1,3}\)\s*$'),
    # Spaced-out math operator expressions: "E [ − ( r − q ) 2 | b q ]"
    re.compile(r'E\s*[\[\(]\s*[−\-]?\s*\(.*?\)\s*[\|].*?[\]\)]'),
    # "max r E [...]" style optimization expressions
    re.compile(r'\b(?:max|min|arg\s*max|arg\s*min|sup|inf)\s+\w\s+E\s*[\[\(]'),
    # N(...) distribution notation: "∼N(0, 1)" or "N ( q, σ 2 q )"
    re.compile(r'[∼~]\s*N\s*\([^)]*\)'),
    # Pr(...) probability notation with math content
    re.compile(r'Pr\s*\([^)]*[≥≤>=<∗σ][^)]*\)'),
    # Var(...) / Cov(...) / E[...] with math content
    re.compile(r'(?:Var|Cov|E)\s*[\[\(][^)\]]*[σ²∗≥≤][^)\]]*[\]\)]'),
    # Spaced-out regression/model equation lines:
    # "y i,t = α i + τ t + ε i,t" — many single-char tokens with = operator
    re.compile(
        r'(?:^|(?<=\.\s))'
        r'[a-zA-Z]\s+[\w,\.]+\s*=\s+'
        r'(?:[a-zA-Z]\s+[\w,\.\(\)]+\s*[+\-]\s*){2,}'
        r'[a-zA-Z][\w,\.\s]*'
    ),
    # ∗ (ASTERISK OPERATOR U+2217) in variable notation: "r ∗", "x ∗ s", "q∗"
    re.compile(r'[A-Za-z]\s*∗\s*[A-Za-z0-9]?'),
    # Isolated superscript/subscript digit clusters with Greek: "σ 2 s", "q 2 i"
    re.compile(r'[αβγδεζηθικλμνξπρστυφχψω]\s+[0-9]\s+[a-zA-Z]'),
    # Leftover math inequality/equality with single-letter variables
    re.compile(r'\b[A-Za-z]\s*[∗_]?\s*[≥≤≈≡≠∼∝]\s*[A-Za-z0-9]'),
]


def is_math_paragraph(text: str) -> bool:
    """Check if an entire paragraph is a math formula (not prose)."""
    stripped = text.strip()
    if not stripped:
        return False

    math_chars = len(re.findall(
        r'[αβγδεζηθικλμνξπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ'
        r'∗∂∇∞∑∏∫∈∉⊂⊃⊆⊇∧∨¬⇒⇔≤≥≈≡≠∼∝±×÷√∅∃∀²³⁴⁵⁶⁷⁸⁹⁰¹=<>^_{}|'
        r'−ˆ˜˙¯†‡§]',  # minus sign, modifier marks, daggers
        stripped
    ))

    total = len(stripped)
    if total == 0:
        return False

    # If >12% of characters are math symbols → formula paragraph
    if math_chars / total > 0.12:
        return True

    # Short text with any math symbols and mostly single-char tokens → equation
    if len(stripped) < 120 and math_chars > 0:
        words = stripped.split()
        single_tokens = sum(1 for w in words if len(w) <= 2)
        if len(words) > 0 and single_tokens / len(words) > 0.5:
            return True

    # Equations marked with trailing equation number "(N)" and an = sign
    if re.search(r'=.*\(\d{1,3}\)\s*$', stripped):
        return True

    return False


# =============================================================================
# TABLE PATTERNS
# =============================================================================

TABLE_PATTERNS = [
    # Lines with multiple consecutive spaces or tabs (table-like alignment)
    re.compile(r'^.*(?:\s{3,}[\d.,]+){2,}.*$', re.MULTILINE),
    # Lines that are mostly numbers separated by spaces (data rows)
    re.compile(r'^[\s]*[\d.,\-()]+(?:\s+[\d.,\-()]+){3,}[\s]*$', re.MULTILINE),
    # Common table headers/labels
    re.compile(r'^\s*(?:Table|TABLE)\s+\d+[.:]\s*.*$', re.MULTILINE),
    # Lines with pipe separators (markdown-style tables)
    re.compile(r'^.*\|.*\|.*\|.*$', re.MULTILINE),
    # Parenthetical standard errors on their own line: (0.023) (0.045) ...
    re.compile(r'^\s*\([\d.]+\)(?:\s+\([\d.]+\))+\s*$', re.MULTILINE),
    # Rows of asterisks with numbers (significance markers)
    re.compile(r'^\s*[\d.,\-]+\*{1,3}(?:\s+[\d.,\-]+\*{1,3})+\s*$', re.MULTILINE),
    # "Notes:" or "Source:" lines often following tables
    re.compile(r'^\s*(?:Notes?|Source|Standard errors?|Robust|Clustered)\s*[:.].*$',
               re.MULTILINE | re.IGNORECASE),
    # Figure/table captions
    re.compile(r'^\s*(?:Figure|FIGURE|Fig\.)\s+\d+[.:]\s*.*$', re.MULTILINE),
]


def is_table_paragraph(text: str) -> bool:
    """Check if an entire paragraph looks like table content."""
    stripped = text.strip()
    if not stripped:
        return False

    words = stripped.split()
    if len(words) < 3:
        return False

    numeric_tokens = sum(1 for w in words if re.match(r'^[\d.,\-+()%*]+$', w))
    if numeric_tokens / len(words) > 0.6:
        return True

    if re.match(r'^\s*\(\([a-z]\)\)', stripped):
        return True

    if re.match(r'^\s*(?:Table|Figure|Fig\.)\s+\d', stripped, re.IGNORECASE):
        return True

    return False


# =============================================================================
# IN-TEXT CITATION PATTERNS
# =============================================================================

CITATION_PATTERNS = [
    # PRIMARY: any (...) containing a 4-digit year (1900-2099).
    # Universal signature of an in-text citation — handles any number of authors,
    # semicolons, compact or spaced forms.
    # [^()]* prevents matching across nested parentheses.
    re.compile(r'\([^()]*(?:19|20)\d{2}[^()]*\)'),

    # NARRATIVE year: Author ( Year ) — remove just the parenthetical year,
    # keeping the author name in the sentence.
    # E.g. "Fukuyama ( 2011 ) notes" → "Fukuyama notes"
    re.compile(r'(?<=\w)\s+\(\s*(?:19|20)\d{2}[a-z]?(?:\s*,\s*(?:19|20)\d{2}[a-z]?)?\s*\)'),
]


# =============================================================================
# FOOTNOTE PATTERNS
# =============================================================================

FOOTNOTE_PATTERNS = [
    # Footnote number at the START of a paragraph/sentence followed by uppercase:
    # "16 There is also..." → "There is also..."
    re.compile(r'^\d{1,3}\s+(?=[A-Z])'),
    # Footnote markers inline: digit right after punctuation, "efforts.1 " or "result,2 "
    re.compile(r'(?<=[a-z.,;:!?])\d{1,2}(?=\s)'),
]


# =============================================================================
# CLEANING FUNCTIONS
# =============================================================================

def remove_math(text: str) -> str:
    """Remove math formulas (LaTeX and broken PDF-extracted)."""
    for pattern in MATH_PATTERNS:
        text = pattern.sub('', text)
    for pattern in BROKEN_MATH_PATTERNS:
        text = pattern.sub('', text)
    return text


def remove_tables(text: str) -> str:
    """Remove table-like content."""
    for pattern in TABLE_PATTERNS:
        text = pattern.sub('', text)
    return text


def remove_citations(text: str) -> str:
    """Remove in-text citations."""
    for pattern in CITATION_PATTERNS:
        text = pattern.sub('', text)
    return text


def remove_footnote_markers(text: str) -> str:
    """Remove footnote number markers from text."""
    for pattern in FOOTNOTE_PATTERNS:
        text = pattern.sub('', text)
    return text


def normalize_text(text: str) -> str:
    """Final text normalization (dehyphenation already done in _pre_normalize)."""
    # Unicode normalize — converts ligatures (ﬁ→fi), superscripts (²→2), etc.
    text = unicodedata.normalize('NFKC', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_paragraph(text: str) -> str:
    """Full cleaning pipeline for a single paragraph."""
    # Step 0: fix PDF artifacts and dehyphenate BEFORE any pattern matching
    text = _pre_normalize(text)

    # Step 1: remove math
    text = remove_math(text)

    # Step 2: remove tables
    text = remove_tables(text)

    # Step 3: remove citations
    text = remove_citations(text)

    # Step 4: remove footnote markers
    text = remove_footnote_markers(text)

    # Step 5: final normalization
    text = normalize_text(text)

    # Step 6: clean up leftover artifacts
    # Orphaned math operators left after formula removal
    text = re.sub(r'\s+[∗≥≤≈≡≠∼∝±×÷√∅∃∀∈∉⊂⊃∧∨]\s+', ' ', text)
    # Empty parentheses / brackets
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    # Leftover double parentheses from figure labels: ((a)), ((b))
    text = re.sub(r'\(\([a-z]\)\)', '', text)
    # Multiple punctuation
    text = re.sub(r'([.,;:])\s*\1+', r'\1', text)
    # Space before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    # Leading punctuation or lone operators at paragraph start
    text = re.sub(r'^\s*[.,;:\-=<>]\s*', '', text)
    # Subsection-style leading numbers: "4.2.1 " at start
    text = re.sub(r'^\d+(?:\.\d+)+\s+', '', text)
    # Collapse whitespace again
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# =============================================================================
# SENTENCE SPLITTING
# =============================================================================

# Split at sentence boundaries: requires a lowercase/digit/bracket before the
# period and a capital letter after the whitespace.  Uses a 2-char fixed-width
# lookbehind so the period STAYS with the preceding sentence.
_SENT_SPLIT = re.compile(r'(?<=[a-z0-9)\"][.!?])\s+(?=[A-Z])')


def split_sentences(text: str) -> list:
    """Split a cleaned paragraph into individual sentences."""
    parts = _SENT_SPLIT.split(text)
    return [s.strip() for s in parts if s.strip() and len(s.strip()) >= 20]


# =============================================================================
# SECTION-LEVEL CLEANING
# =============================================================================

def _merge_broken_paragraphs(paragraphs: list) -> list:
    """Merge consecutive paragraphs that were broken mid-sentence.

    If a paragraph does not end with sentence-ending punctuation (.!?:)
    it was likely split at a page or column boundary — merge it with the
    next paragraph so the sentence splitter receives complete sentences.
    """
    if not paragraphs:
        return []
    merged = [paragraphs[0]]
    for para in paragraphs[1:]:
        prev = merged[-1].rstrip()
        # Previous chunk doesn't end with sentence-ending punctuation → merge
        if prev and not re.search(r'[.!?:]["\u201d]?\s*$', prev):
            merged[-1] = merged[-1] + ' ' + para
        else:
            merged.append(para)
    return merged


def clean_sections(sections: list) -> list:
    """Clean all paragraphs in a sections list.
    Each output chunk is a single complete sentence.
    Drops paragraphs that are pure math or tables, or become too short after cleaning.
    """
    skip_sections = {'acknowledgements', 'acknowledgments'}
    cleaned = []
    for section in sections:
        # Clean section name (fix diacritics, dehyphenate trailing hyphens)
        section_name = _pre_normalize(section["section_name"])
        section_name = re.sub(r'-\s*$', '', section_name)
        if section_name.lower().strip().rstrip(':') in skip_sections:
            continue

        # ── Step 1: filter out math/table paragraphs ──
        prose_paras = []
        for para in section["paragraphs"]:
            para_norm = _pre_normalize(para)
            if is_math_paragraph(para_norm):
                continue
            if is_table_paragraph(para_norm):
                continue
            prose_paras.append(para)

        if not prose_paras:
            continue

        # ── Step 2: heal mid-sentence breaks ──
        prose_paras = _merge_broken_paragraphs(prose_paras)

        # ── Step 3: clean each merged block and split into sentences ──
        new_paras = []
        for para in prose_paras:
            cleaned_text = clean_paragraph(para)
            if cleaned_text and len(cleaned_text) >= 20:
                new_paras.extend(split_sentences(cleaned_text))

        if new_paras:
            cleaned.append({
                "section_name": section_name,
                "paragraphs": new_paras,
            })

    return cleaned
