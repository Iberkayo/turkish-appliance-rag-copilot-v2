"""
Page-level PDF extraction pipeline.

Parses every PDF listed in the corpus manifest page-by-page and writes:
- data/processed/pages.jsonl
- data/eval/page_audit.xlsx

No chunking or embedding is performed.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import fitz  # pymupdf
import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MANIFEST_PATH = Path("data/raw/corpus_manifest.xlsx")
PAGES_JSONL_PATH = Path("data/processed/pages.jsonl")
PAGE_AUDIT_PATH = Path("data/eval/page_audit.xlsx")

TR_TRANSLATION = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> pd.DataFrame:
    """Read the corpus manifest Excel file."""
    logger.info("Loading manifest from %s", path)
    df = pd.read_excel(path)
    required = {"brand", "category", "file_name", "local_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    return df


def normalize_for_doc_id(text: str) -> str:
    """Normalize a string for use in a doc_id: Turkish chars -> ASCII, alphanumeric + underscore only."""
    text = text.translate(TR_TRANSLATION)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text.lower()


def build_doc_id(brand: str, category: str, file_name: str) -> str:
    """Build a normalized doc_id from brand, category and file_name stem."""
    stem = Path(file_name).stem
    parts = [brand, category, stem]
    normalized = "_".join(normalize_for_doc_id(part) for part in parts)
    return normalized


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def clean_text(text: str) -> str:
    """Clean extracted page text."""
    # strip control characters that break downstream writers (Excel, JSONL parsers, etc.)
    text = _CONTROL_CHAR_RE.sub("", text)
    # collapse multiple spaces
    text = re.sub(r" +", " ", text)
    # normalize excessive newlines (more than 2 -> 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(row: pd.Series) -> list[dict[str, Any]]:
    """Extract pages from a single PDF and return a list of page records."""
    file_path = Path(row["local_path"])
    brand = row["brand"]
    category = row["category"]
    file_name = row["file_name"]
    doc_id = build_doc_id(brand, category, file_name)

    pages: list[dict[str, Any]] = []

    try:
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        with fitz.open(str(file_path)) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                raw_text = page.get_text()
                text = clean_text(raw_text)
                char_count = len(text)
                word_count = len(text.split())

                pages.append(
                    {
                        "doc_id": doc_id,
                        "brand": brand,
                        "category": category,
                        "file_name": file_name,
                        "local_path": str(file_path),
                        "page_number": page_num + 1,
                        "text": text,
                        "char_count": char_count,
                        "word_count": word_count,
                    }
                )

    except Exception as exc:
        logger.error("Failed to process %s: %s", file_name, exc)
        # Emit a single fallback record so the failure is visible
        pages.append(
            {
                "doc_id": doc_id,
                "brand": brand,
                "category": category,
                "file_name": file_name,
                "local_path": str(file_path),
                "page_number": 0,
                "text": "",
                "char_count": 0,
                "word_count": 0,
            }
        )

    return pages


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Written %d page records to %s", len(records), path)


def build_audit_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build an audit DataFrame from page records."""
    df = pd.DataFrame(records)
    df["text_preview"] = df["text"].str[:200]
    audit_cols = [
        "doc_id",
        "brand",
        "category",
        "file_name",
        "page_number",
        "char_count",
        "word_count",
        "text_preview",
    ]
    return df[audit_cols]


def print_summary(
    manifest: pd.DataFrame,
    audit_df: pd.DataFrame,
    all_records: list[dict[str, Any]],
) -> None:
    """Print a terminal summary of the extraction results."""
    total_pdfs = len(manifest)
    total_pages = len(audit_df)
    total_words = int(audit_df["word_count"].sum())
    empty_pages = int((audit_df["char_count"] == 0).sum())
    short_pages = int((audit_df["char_count"] < 100).sum())

    cat_pages = audit_df.groupby("category").size().sort_index()
    cat_words = audit_df.groupby("category")["word_count"].sum().sort_index()

    print("\n" + "=" * 60)
    print("PAGE-LEVEL EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total PDFs        : {total_pdfs}")
    print(f"Total pages       : {total_pages}")
    print(f"Total words       : {total_words:,}")
    print(f"Empty pages       : {empty_pages}")
    print(f"Short pages (<100): {short_pages}")
    print("=" * 60)

    print("\n--- Category page counts ---")
    for category, count in cat_pages.items():
        print(f"  {category}: {count}")

    print("\n--- Category word counts ---")
    for category, count in cat_words.items():
        print(f"  {category}: {count:,.0f}")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    logger.info("Found %d PDF entries in manifest", len(manifest))

    all_records: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        pages = extract_pdf_pages(row)
        all_records.extend(pages)
        logger.info(
            "Extracted %d pages from %s", len(pages), row["file_name"]
        )

    # Write JSONL
    write_jsonl(all_records, PAGES_JSONL_PATH)

    # Build and write audit Excel
    audit_df = build_audit_df(all_records)
    PAGE_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_excel(PAGE_AUDIT_PATH, index=False)
    logger.info("Page audit saved to %s", PAGE_AUDIT_PATH)

    # Terminal summary
    print_summary(manifest, audit_df, all_records)


if __name__ == "__main__":
    main()
