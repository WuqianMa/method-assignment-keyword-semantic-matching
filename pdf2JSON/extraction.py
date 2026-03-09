#!/usr/bin/env python3
"""
Raw text extraction from PDF and DOCX files.
Produces structured sections with paragraphs — no heavy cleaning applied here.
"""

import fitz  # pymupdf
import docx
import re
import logging
from pathlib import Path
from collections import Counter

logger = logging.getLogger(__name__)

# Known academic section header patterns
KNOWN_HEADERS_RE = re.compile(
    r"^(?:[\dIVXivx]+[.\s)]+\s*)?"  # optional numbering
    r"(Introduction|Literature\s+Review|Related\s+(?:Work|Literature)|"
    r"Theoretical\s+(?:Framework|Background|Model)|Background|"
    r"Model|The\s+Model|Data(?:\s+and\s+Method(?:ology|s)?)?|"
    r"Data\s+Description|Sample|"
    r"Method(?:ology|s)?|Empirical\s+(?:Strategy|Framework|Analysis|Model|Specification)|"
    r"Identification(?:\s+Strategy)?|Estimation(?:\s+Strategy)?|"
    r"Research\s+Design|Experimental\s+Design|"
    r"Results?|Findings|Empirical\s+Results?|Main\s+Results?|"
    r"Discussion|Analysis|Robustness(?:\s+Checks?)?|"
    r"Extensions?|Heterogeneity|Mechanisms?|"
    r"Conclusion(?:s|\s+and\s+(?:Discussion|Policy\s+Implications))?|"
    r"Summary(?:\s+and\s+Conclusion)?|Concluding\s+Remarks|"
    r"Policy\s+Implications|"
    r"References?|Bibliography|Works?\s+Cited|"
    r"Appendix(?:\s+[A-Z])?|Online\s+Appendix|"
    r"Abstract|Keywords?|JEL\s+Classification"
    r")\s*:?\s*$",
    re.IGNORECASE
)

REFERENCES_HEADERS = {
    'references', 'bibliography', 'works cited', 'reference', 'literature cited'
}

APPENDIX_HEADERS = {
    'appendix', 'online appendix', 'supplementary material',
    'supplementary appendix', 'supplemental material',
}


# =============================================================================
# PDF EXTRACTOR
# =============================================================================

class PDFExtractor:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.doc = fitz.open(str(filepath))
        self.warnings = []

    def extract(self) -> dict:
        """Returns {sections: [...], format_class: str, warnings: []}"""
        try:
            if self._is_slides():
                return self._extract_slides()

            spans = self._extract_spans()
            if not spans:
                self.warnings.append("No text extracted (possibly scanned/image PDF)")
                return {"sections": [], "format_class": "abstract_only",
                        "warnings": self.warnings}

            body_size = self._detect_body_font_size(spans)
            noise_texts = self._detect_headers_footers(spans)
            spans = [s for s in spans if s['norm_text'] not in noise_texts]
            headers = self._find_section_headers(spans, body_size)
            refs_idx = self._find_references_start(headers)
            appendix_idx = self._find_appendix_start(headers)

            # Cut at whichever comes first: references or appendix
            cut_idx = None
            if refs_idx is not None and appendix_idx is not None:
                cut_idx = min(refs_idx, appendix_idx)
            elif refs_idx is not None:
                cut_idx = refs_idx
            elif appendix_idx is not None:
                cut_idx = appendix_idx

            sections = self._build_sections(spans, headers, cut_idx, body_size)
            format_class = self._classify_format(headers, sections)

            return {"sections": sections, "format_class": format_class,
                    "warnings": self.warnings}
        finally:
            self.doc.close()

    def _is_slides(self) -> bool:
        if self.doc.page_count == 0:
            return False
        page = self.doc[0]
        w, h = page.rect.width, page.rect.height
        if w > h:
            return True
        total_chars = sum(len(p.get_text()) for p in self.doc)
        avg = total_chars / self.doc.page_count if self.doc.page_count > 0 else 0
        if avg < 200:
            return True
        if 'slide' in self.filepath.stem.lower():
            return True
        return False

    def _extract_spans(self) -> list:
        """Extract text as merged lines with font metadata."""
        all_lines = []
        for page_num, page in enumerate(self.doc):
            page_dict = page.get_text("dict")
            page_height = page.rect.height
            for block in page_dict["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    line_text_parts = []
                    bold_chars = 0
                    total_chars = 0
                    size_chars = Counter()
                    line_bbox = list(line["bbox"])

                    for span in line["spans"]:
                        text = span["text"]
                        if not text.strip():
                            continue
                        line_text_parts.append(text)
                        n = len(text.strip())
                        total_chars += n
                        sz = round(span["size"], 1)
                        size_chars[sz] += n
                        if span["flags"] & (1 << 4):
                            bold_chars += n

                    if not line_text_parts or total_chars == 0:
                        continue

                    merged_text = ' '.join(line_text_parts).strip()
                    if not merged_text:
                        continue

                    dom_size = size_chars.most_common(1)[0][0]
                    is_bold = (bold_chars / total_chars) > 0.5

                    all_lines.append({
                        "text": merged_text,
                        "norm_text": merged_text.lower().strip(),
                        "size": dom_size,
                        "is_bold": is_bold,
                        "bbox": tuple(line_bbox),
                        "page": page_num,
                        "page_height": page_height,
                        "block_bbox": block["bbox"],
                    })
        return all_lines

    def _detect_body_font_size(self, spans: list) -> float:
        size_chars = Counter()
        for s in spans:
            size_chars[s["size"]] += len(s["text"])
        if not size_chars:
            return 10.0
        return size_chars.most_common(1)[0][0]

    def _detect_headers_footers(self, spans: list) -> set:
        top_pages = {}
        bottom_pages = {}

        for s in spans:
            ph = s["page_height"]
            if ph <= 0:
                continue
            y_ratio = s["bbox"][1] / ph
            norm = s["norm_text"]
            if len(norm) < 3:
                continue

            if y_ratio < 0.10:
                top_pages.setdefault(norm, set()).add(s["page"])
            elif y_ratio > 0.90:
                bottom_pages.setdefault(norm, set()).add(s["page"])

        noise = set()
        for text, pages in top_pages.items():
            if len(pages) >= 3:
                noise.add(text)
        for text, pages in bottom_pages.items():
            if len(pages) >= 3:
                noise.add(text)

        for s in spans:
            if re.match(r'^\d{1,4}$', s["text"].strip()):
                ph = s["page_height"]
                if ph <= 0:
                    continue
                y_ratio = s["bbox"][1] / ph
                if y_ratio > 0.90 or y_ratio < 0.08:
                    noise.add(s["norm_text"])

        return noise

    def _find_section_headers(self, spans: list, body_size: float) -> list:
        headers = []
        for i, s in enumerate(spans):
            text = s["text"].strip()
            size = s["size"]
            is_bold = s["is_bold"]

            if KNOWN_HEADERS_RE.match(text):
                if size >= body_size + 1.0:
                    cleaned = re.sub(r'^[\dIVXivx]+[.\s)]+\s*', '', text).strip()
                    headers.append((i, cleaned))
                elif is_bold and size >= body_size - 0.5:
                    cleaned = re.sub(r'^[\dIVXivx]+[.\s)]+\s*', '', text).strip()
                    headers.append((i, cleaned))
                continue

            if not is_bold:
                continue
            if size < body_size + 1.5:
                continue
            if len(text) < 3 or len(text) > 80:
                continue
            if not text[0].isupper() and not text[0].isdigit():
                continue
            word_count = len(text.split())
            if word_count < 2:
                single_word_ok = text.lower().rstrip(':') in {
                    'background', 'model', 'data', 'methodology', 'methods',
                    'results', 'findings', 'discussion', 'analysis',
                    'extensions', 'heterogeneity', 'mechanisms', 'estimation',
                    'identification', 'sample', 'covariates', 'treatment',
                }
                if not single_word_ok:
                    continue

            headers.append((i, text))

        # Merge consecutive header lines
        merged_headers = []
        for span_idx, name in headers:
            if (merged_headers
                    and span_idx == merged_headers[-1][0] + 1
                    and spans[span_idx]["page"] == spans[merged_headers[-1][0]]["page"]
                    and abs(spans[span_idx]["size"] - spans[merged_headers[-1][0]]["size"]) < 0.5):
                prev_idx, prev_name = merged_headers[-1]
                if prev_name.endswith('-'):
                    combined = prev_name[:-1] + name
                else:
                    combined = prev_name + ' ' + name
                merged_headers[-1] = (prev_idx, combined)
            else:
                merged_headers.append((span_idx, name))

        return merged_headers

    def _find_references_start(self, headers: list):
        for span_idx, name in headers:
            if name.lower().strip().rstrip(':') in REFERENCES_HEADERS:
                return span_idx
        return None

    def _find_appendix_start(self, headers: list):
        for span_idx, name in headers:
            name_lower = name.lower().strip().rstrip(':')
            if any(name_lower.startswith(a) for a in APPENDIX_HEADERS):
                return span_idx
        return None

    def _build_sections(self, spans, headers, cut_idx, body_size) -> list:
        if not spans:
            return []

        if cut_idx is not None:
            spans = spans[:cut_idx]
            headers = [(idx, name) for idx, name in headers
                       if idx < cut_idx
                       and name.lower().strip().rstrip(':') not in REFERENCES_HEADERS
                       and not any(name.lower().strip().rstrip(':').startswith(a)
                                   for a in APPENDIX_HEADERS)]

        if not headers:
            paragraphs = self._spans_to_paragraphs(spans, body_size)
            if not paragraphs:
                return []
            sections = []
            chunk_size = 10
            for i in range(0, len(paragraphs), chunk_size):
                chunk = paragraphs[i:i + chunk_size]
                section_n = (i // chunk_size) + 1
                sections.append({
                    "section_name": f"Unknown_{section_n}",
                    "paragraphs": chunk
                })
            return sections

        sections = []
        first_header_idx = headers[0][0]
        pre_spans = spans[:first_header_idx]
        if pre_spans:
            pre_paras = self._spans_to_paragraphs(pre_spans, body_size)
            if pre_paras:
                # Label as "Abstract" if any paragraph starts with that word
                has_abstract = any(
                    p.strip().lower().startswith('abstract')
                    for p in pre_paras
                )
                sections.append({
                    "section_name": "Abstract" if has_abstract else "Preamble",
                    "paragraphs": pre_paras
                })

        for h in range(len(headers)):
            start_idx = headers[h][0] + 1
            end_idx = headers[h + 1][0] if h + 1 < len(headers) else len(spans)
            section_spans = spans[start_idx:end_idx]
            paras = self._spans_to_paragraphs(section_spans, body_size)
            sections.append({
                "section_name": headers[h][1],
                "paragraphs": paras
            })

        return sections

    def _spans_to_paragraphs(self, spans: list, body_size: float) -> list:
        if not spans:
            return []

        paragraphs = []
        current_texts = []
        prev_block_bottom = None
        prev_page = None
        avg_line_height = body_size * 1.4

        def _flush():
            para_text = ' '.join(current_texts)
            if para_text.strip() and len(para_text.strip()) >= 20:
                paragraphs.append(para_text.strip())
            current_texts.clear()

        def _ends_sentence(texts: list) -> bool:
            """True if the accumulated text ends at a sentence boundary."""
            joined = ' '.join(texts).rstrip()
            return bool(joined) and joined[-1] in '.!?:'

        for s in spans:
            block_top = s["block_bbox"][1]
            block_bottom = s["block_bbox"][3]
            cur_page = s["page"]

            if prev_block_bottom is not None:
                page_changed = (cur_page != prev_page)
                gap = block_top - prev_block_bottom if not page_changed else 0

                wants_break = (
                    (not page_changed and gap > avg_line_height * 0.8)
                    or page_changed
                )
                # Only flush at an actual sentence boundary — never mid-sentence
                if wants_break and _ends_sentence(current_texts):
                    _flush()

            current_texts.append(s["text"])
            prev_block_bottom = block_bottom
            prev_page = cur_page

        if current_texts:
            _flush()

        return paragraphs

    def _classify_format(self, headers, sections) -> str:
        header_names = {name.lower() for _, name in headers}
        has_intro = any('introduction' in h for h in header_names)
        has_conclusion = any('conclusion' in h or 'summary' in h
                            for h in header_names)
        total_paras = sum(len(s["paragraphs"]) for s in sections)

        if not headers:
            if self.doc.page_count <= 3 or total_paras <= 5:
                return "abstract_only"
            return "no_section_headers"
        if has_intro and has_conclusion:
            return "standard_complete"
        return "partial_sections"

    def _extract_slides(self) -> dict:
        self.warnings.append("Slide-format PDF detected")
        sections = []
        for i, page in enumerate(self.doc):
            text = page.get_text().strip()
            if text and len(text) >= 10:
                sections.append({
                    "section_name": f"Slide_{i + 1}",
                    "paragraphs": [text]
                })
        return {"sections": sections, "format_class": "slides",
                "warnings": self.warnings}


# =============================================================================
# DOCX EXTRACTOR
# =============================================================================

class DOCXExtractor:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.warnings = []

    def extract(self) -> dict:
        doc = docx.Document(str(self.filepath))
        sections = []
        current_section = {"section_name": "Unknown_1", "paragraphs": []}
        stop_processing = False

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            is_heading = para.style.name.startswith("Heading")
            if not is_heading and para.runs and para.runs[0].bold and len(text) < 80:
                is_heading = True

            if is_heading:
                text_lower = text.lower().strip().rstrip(':')
                # Stop at references or appendix
                if text_lower in REFERENCES_HEADERS:
                    stop_processing = True
                    break
                if any(text_lower.startswith(a) for a in APPENDIX_HEADERS):
                    stop_processing = True
                    break

                if current_section["paragraphs"]:
                    sections.append(current_section)

                if KNOWN_HEADERS_RE.match(text):
                    name = re.sub(r'^[\dIVXivx]+[.\s)]+\s*', '', text).strip()
                else:
                    name = text
                current_section = {"section_name": name, "paragraphs": []}
            else:
                if stop_processing:
                    break
                if text and len(text) >= 20:
                    current_section["paragraphs"].append(text)

        if current_section["paragraphs"]:
            sections.append(current_section)

        # Classify
        if all(s["section_name"].startswith("Unknown_") for s in sections):
            format_class = "no_section_headers"
        elif not sections or (len(sections) == 1
                              and len(sections[0]["paragraphs"]) <= 5):
            format_class = "abstract_only"
        else:
            header_names = {s["section_name"].lower() for s in sections}
            has_intro = any('introduction' in h for h in header_names)
            has_conclusion = any('conclusion' in h or 'summary' in h
                                for h in header_names)
            if has_intro and has_conclusion:
                format_class = "standard_complete"
            else:
                format_class = "partial_sections"

        return {"sections": sections, "format_class": format_class,
                "warnings": self.warnings}


def extract_paper(filepath: Path) -> dict:
    """Unified entry point. Returns raw extraction result."""
    file_type = filepath.suffix.lower().lstrip('.')
    if file_type == "pdf":
        return PDFExtractor(filepath).extract()
    elif file_type == "docx":
        return DOCXExtractor(filepath).extract()
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
