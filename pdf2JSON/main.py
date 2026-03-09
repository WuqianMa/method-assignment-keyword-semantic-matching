#!/usr/bin/env python3
"""
Main entry point for the paper parsing pipeline.

Usage:
    python main.py                    # Full pipeline: extract + clean all papers
    python main.py --extract          # Step 1 only: extract raw JSON to middle/raw/
    python main.py --clean            # Step 2 only: clean raw JSON from middle/raw/ to output/
    python main.py --file paper.pdf   # Process a single file (full pipeline)
"""

import argparse
import sys
import io
import logging
from pathlib import Path
from tqdm import tqdm

from pipeline import (
    INPUT_DIR, RAW_DIR, OUTPUT_DIR,
    extract_single, save_json, load_json,
    clean_and_flatten,
)

LOG_FILE = "parse_papers.log"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def get_paper_files() -> list:
    """Get all PDF/DOCX files from input directory."""
    files = []
    for ext in ('*.pdf', '*.docx'):
        files.extend(INPUT_DIR.glob(ext))
    return sorted(files)


def json_name(filepath: Path) -> str:
    """Same name as the source file but with .json extension."""
    return filepath.stem + ".json"


def step_extract(files: list):
    """Step 1: Extract raw content from PDFs/DOCXs into middle/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n--- STEP 1: EXTRACT ({len(files)} files) ---")
    for filepath in tqdm(files, desc="Extracting"):
        out_path = RAW_DIR / json_name(filepath)
        try:
            raw = extract_single(filepath)
            save_json(raw, out_path)
        except Exception as e:
            logging.error(f"Failed to extract {filepath.name}: {e}")
    print(f"Raw JSON saved to {RAW_DIR}/")


def step_clean(raw_files: list = None):
    """Step 2: Clean raw JSON from middle/raw/ and save to output/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if raw_files is None:
        raw_files = sorted(RAW_DIR.glob("*.json"))

    print(f"\n--- STEP 2: CLEAN ({len(raw_files)} files) ---")
    for raw_path in tqdm(raw_files, desc="Cleaning"):
        try:
            raw = load_json(raw_path)
            rows = clean_and_flatten(raw)
            save_json(rows, OUTPUT_DIR / raw_path.name)
        except Exception as e:
            logging.error(f"Failed to clean {raw_path.name}: {e}")
    print(f"Cleaned JSON saved to {OUTPUT_DIR}/")


def main():
    # Fix Windows console encoding
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    setup_logging()

    parser = argparse.ArgumentParser(description="Paper parsing pipeline")
    parser.add_argument('--extract', action='store_true',
                        help='Step 1 only: extract raw JSON to middle/raw/')
    parser.add_argument('--clean', action='store_true',
                        help='Step 2 only: clean raw JSON from middle/raw/ to output/')
    parser.add_argument('--file', type=str,
                        help='Process a single file (full pipeline)')
    args = parser.parse_args()

    if args.file:
        # Single file mode
        filepath = INPUT_DIR / args.file
        if not filepath.exists():
            filepath = Path(args.file)
        if not filepath.exists():
            print(f"ERROR: File not found: {args.file}")
            sys.exit(1)

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        out_name = json_name(filepath)
        print(f"\nProcessing: {filepath.name}")

        raw = extract_single(filepath)
        save_json(raw, RAW_DIR / out_name)
        print(f"  Raw -> {RAW_DIR / out_name}")

        rows = clean_and_flatten(raw)
        save_json(rows, OUTPUT_DIR / out_name)
        print(f"  Clean -> {OUTPUT_DIR / out_name}")
        print(f"  Format: {raw['format_class']}, Paragraphs: {len(rows)}")
        return

    if args.extract:
        files = get_paper_files()
        step_extract(files)
    elif args.clean:
        step_clean()
    else:
        # Full pipeline
        files = get_paper_files()
        step_extract(files)
        step_clean()


if __name__ == "__main__":
    main()
