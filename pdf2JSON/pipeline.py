#!/usr/bin/env python3
"""
Pipeline orchestration: extract, clean, and flatten papers into JSON rows.
No metadata — filename is the key for future matching.
"""

import json
import logging
from pathlib import Path

from extraction import extract_paper
from cleaning import clean_sections

logger = logging.getLogger(__name__)

# =============================================================================
# PATHS
# =============================================================================

INPUT_DIR = Path("input")
RAW_DIR = Path("middle/raw")
OUTPUT_DIR = Path("output")


# =============================================================================
# STEP 1: EXTRACT (PDF/DOCX -> raw JSON in middle/raw/)
# =============================================================================

def extract_single(filepath: Path) -> dict:
    """Extract raw content from a single paper. Returns a raw result dict."""
    file_name = filepath.name

    try:
        result = extract_paper(filepath)
    except Exception as e:
        logger.error(f"Extraction failed for {file_name}: {e}")
        result = {
            "sections": [],
            "format_class": "error",
            "warnings": [f"extraction_failed: {e}"],
        }

    return {
        "file_name": file_name,
        "format_class": result["format_class"],
        "warnings": result.get("warnings", []),
        "sections": result["sections"],
    }


def save_json(data, output_path: Path):
    """Save dict/list to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path):
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# =============================================================================
# STEP 2: CLEAN (raw JSON -> cleaned flat JSON in output/)
# =============================================================================

def clean_and_flatten(raw_result: dict) -> list:
    """Clean sections and flatten into rows (one per paragraph)."""
    file_name = raw_result["file_name"]
    format_class = raw_result["format_class"]
    warnings_str = "; ".join(raw_result.get("warnings", []))

    # Apply cleaning
    cleaned_sections = clean_sections(raw_result["sections"])

    # Flatten
    rows = []
    section_id = 0
    for section in cleaned_sections:
        for para_id, para_text in enumerate(section["paragraphs"]):
            rows.append({
                "file_name": file_name,
                "format_class": format_class,
                "section_id": section_id,
                "section_name": section["section_name"],
                "paragraph_id": para_id,
                "text": para_text,
                "warnings": warnings_str,
            })
        section_id += 1

    if not rows:
        rows.append({
            "file_name": file_name,
            "format_class": format_class,
            "section_id": 0,
            "section_name": "Empty",
            "paragraph_id": 0,
            "text": "",
            "warnings": warnings_str or "no_text_extracted",
        })

    return rows
