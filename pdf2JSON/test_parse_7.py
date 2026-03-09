#!/usr/bin/env python3
"""Quick test: run pipeline on 7 representative edge-case papers."""

import json
import sys
import io
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pipeline import (
    INPUT_DIR, RAW_DIR, OUTPUT_DIR,
    extract_single, save_json,
    clean_and_flatten,
)

TEST_FILES = [
    "baragwanath-2025-01-14_9599.pdf",   # standard complete
    "trinh-2025-01-31_7467.pdf",          # abstract-only
    "kim-2025-01-29_5963.pdf",            # tiny, no headers
    "Simona_slides_2025_SIOE.pdf",        # slides
    "avery-2025-01-01_9752.docx",         # DOCX
    "sng-2025-03-23_4611.pdf",            # no section headers
    "supervisor_accuracy.pdf",            # non-standard name
]


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    for fname in TEST_FILES:
        filepath = INPUT_DIR / fname
        if not filepath.exists():
            print(f"MISSING: {fname}")
            continue

        print(f"{'='*70}")
        print(f"FILE: {fname}")
        print(f"SIZE: {filepath.stat().st_size / 1024:.1f} KB")

        # Step 1: Extract raw
        raw = extract_single(filepath)
        json_name = filepath.stem + ".json"
        save_json(raw, RAW_DIR / json_name)

        # Step 2: Clean and flatten
        rows = clean_and_flatten(raw)
        save_json(rows, OUTPUT_DIR / json_name)

        # Summary
        format_class = rows[0]["format_class"] if rows else "?"
        sections = set()
        for r in rows:
            sections.add((r["section_id"], r["section_name"]))
        n_paras = len(rows)
        warnings = rows[0].get("warnings", "") if rows else ""

        print(f"FORMAT: {format_class}")
        print(f"PARAGRAPHS: {n_paras}")
        print(f"SECTIONS ({len(sections)}):")
        for sid, sname in sorted(sections):
            count = sum(1 for r in rows if r["section_id"] == sid)
            print(f"  [{sid}] {sname} ({count} paragraphs)")
        if warnings:
            print(f"WARNINGS: {warnings}")

        # Show first paragraph text preview
        if rows and rows[0]["text"]:
            preview = rows[0]["text"][:150]
            print(f"FIRST PARA: {preview}...")
        print()


if __name__ == "__main__":
    main()
