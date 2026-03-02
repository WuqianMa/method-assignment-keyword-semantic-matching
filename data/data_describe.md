# Data Description: Academic Papers Dataset

This directory contains processed metadata and normalized text content from academic papers. The data is split between high-level paper information and granular, paragraph-level text.

## Directory Structure

* `metadata/meta_data.json`: A master list of all papers.
* `papers_json_normalize/`: A directory containing individual JSON files for each paper, named using the format `{author}-{date}-{id}.json`.

---

## 1. Metadata Schema (`meta_data.json`)

This file is a list of objects, where each object represents one paper. Use this to find the `paper_id` or the `file_name` for a specific study.

| Key | Description |
| --- | --- |
| `paper_id` | **Primary Key.** Unique integer identifying the paper. |
| `file_name` | The original PDF filename (matches the JSON names in the normalize folder). |
| `title` | Full title of the paper. |
| `abstract` | Brief summary of the research. |
| `author` | Lead author name. |
| `university` | Affiliated institution. |
| `url` | Source link to the paper. |

---

## 2. Normalized Text Schema (`papers_json_normalize/*.json`)

Each file in this folder contains the full text of a single paper, broken down into paragraph-level objects for granular retrieval.

| Key | Description |
| --- | --- |
| `paper_id` | Links back to the `paper_id` in `meta_data.json`. |
| `section_id` | Integer index of the section. |
| `section_name` | The header title (e.g., "Introduction", "Methodology"). |
| `paragraph_id` | Sequential index of the paragraph within that section. |
| `text` | The actual string content of the paragraph. |
| `extraction_warnings` | Flags any issues encountered during the PDF-to-JSON conversion. |

---

## Data Relationships & Joining

To find the full text for a specific author:

1. Search `meta_data.json` for the `author` or `title`.
2. Note the `file_name` (e.g., `alfitian-2025-05-09_204.pdf`).
3. Open the corresponding JSON in `papers_json_normalize/` (e.g., `alfitian-2025-05-09_204.json`).

> **Note:** The `paper_id` is the reliable foreign key between the metadata and the normalized text chunks.

