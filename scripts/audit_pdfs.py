"""
PDF Corpus Audit System

Parses every PDF listed in the corpus manifest and produces a quality audit
report.  No chunking or embedding is performed — this step is purely for
corpus-quality analysis.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pymupdf

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
OUTPUT_PATH = Path("data/eval/pdf_audit.xlsx")

MODEL_PATTERNS = [
    re.compile(r"[A-Z]{2,5}\d{4,6}"),
    re.compile(r"[A-Z0-9\-]{5,20}"),
]

WARNING_PICTURE_OMITTED_THRESHOLD = 5
WARNING_BLANK_LINE_RATIO = 0.30
WARNING_MIN_CHAR_COUNT = 500

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


def extract_model(text: str) -> str:
    """Attempt to extract a model number from the parsed text.

    Matches must contain at least one digit to avoid common words
    such as 'KULLANIM' being flagged as model numbers.
    """
    for pattern in MODEL_PATTERNS:
        for match in pattern.finditer(text):
            candidate = match.group(0)
            if any(ch.isdigit() for ch in candidate):
                return candidate
    return ""


def check_warnings(text: str, char_count: int) -> list[str]:
    """Return a list of quality warnings for the parsed text."""
    warnings: list[str] = []

    if char_count < WARNING_MIN_CHAR_COUNT:
        warnings.append(f"char_count < {WARNING_MIN_CHAR_COUNT}")

    omitted_count = text.lower().count("picture omitted")
    if omitted_count > WARNING_PICTURE_OMITTED_THRESHOLD:
        warnings.append(f"{omitted_count} 'picture omitted' occurrences")

    lines = text.splitlines()
    if lines:
        blank_lines = sum(1 for line in lines if line.strip() == "")
        ratio = blank_lines / len(lines)
        if ratio > WARNING_BLANK_LINE_RATIO:
            warnings.append(f"blank line ratio {ratio:.1%}")

    return warnings


def audit_single_pdf(row: pd.Series) -> dict[str, Any]:
    """Audit a single PDF and return a dictionary of metrics."""
    file_path = Path(row["local_path"])
    brand = row["brand"]
    category = row["category"]
    file_name = row["file_name"]

    record: dict[str, Any] = {
        "brand": brand,
        "category": category,
        "file_name": file_name,
        "page_count": 0,
        "char_count": 0,
        "word_count": 0,
        "detected_model": "",
        "parse_status": "pending",
        "preview": "",
        "warnings": "",
    }

    try:
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        # Open with pymupdf to get page count and extract text
        with pymupdf.open(str(file_path)) as doc:
            record["page_count"] = len(doc)
            text_parts: list[str] = []
            for page in doc:
                text_parts.append(page.get_text())
            text = "\n".join(text_parts)

        record["char_count"] = len(text)
        record["word_count"] = len(text.split())
        record["preview"] = text[:300]
        record["detected_model"] = extract_model(text)
        record["parse_status"] = "success"

        warnings = check_warnings(text, record["char_count"])
        if warnings:
            record["warnings"] = "; ".join(warnings)
            logger.warning("Warnings for %s: %s", file_name, record["warnings"])

    except Exception as exc:
        record["parse_status"] = "failed"
        record["warnings"] = str(exc)
        logger.error("Failed to parse %s: %s", file_name, exc)

    return record


def print_summary(df: pd.DataFrame) -> None:
    """Print a terminal summary of the audit results."""
    total = len(df)
    success = (df["parse_status"] == "success").sum()
    failed = (df["parse_status"] == "failed").sum()
    avg_pages = df["page_count"].mean()
    avg_words = df["word_count"].mean()

    print("\n" + "=" * 60)
    print("PDF AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total PDFs       : {total}")
    print(f"Successful parses: {success}")
    print(f"Failed parses    : {failed}")
    print(f"Avg page count   : {avg_pages:.1f}")
    print(f"Avg word count   : {avg_words:.0f}")
    print("=" * 60)

    # Smallest 5 PDFs by word count
    print("\n--- Smallest 5 PDFs (by word count) ---")
    smallest = df.nsmallest(5, "word_count")[
        ["file_name", "category", "word_count", "parse_status"]
    ]
    print(smallest.to_string(index=False))

    # Largest 5 PDFs by word count
    print("\n--- Largest 5 PDFs (by word count) ---")
    largest = df.nlargest(5, "word_count")[
        ["file_name", "category", "word_count", "parse_status"]
    ]
    print(largest.to_string(index=False))

    # Warnings summary
    warned = df[df["warnings"] != ""]
    if not warned.empty:
        print(f"\n--- PDFs with warnings ({len(warned)}) ---")
        print(
            warned[["file_name", "category", "warnings"]].to_string(index=False)
        )
    else:
        print("\n--- No warnings ---")

    print("\n" + "=" * 60)
    print("Sample preview (first 3 rows)")
    print("=" * 60)
    preview_cols = [
        "file_name",
        "page_count",
        "word_count",
        "detected_model",
        "parse_status",
    ]
    print(df[preview_cols].head(3).to_string(index=False))
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    logger.info("Found %d PDF entries in manifest", len(manifest))

    records: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        record = audit_single_pdf(row)
        records.append(record)

    audit_df = pd.DataFrame(records)

    # Reorder columns for readability
    column_order = [
        "brand",
        "category",
        "file_name",
        "page_count",
        "char_count",
        "word_count",
        "detected_model",
        "parse_status",
        "preview",
        "warnings",
    ]
    audit_df = audit_df[column_order]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_excel(OUTPUT_PATH, index=False)
    logger.info("Audit report saved to %s", OUTPUT_PATH)

    print_summary(audit_df)


if __name__ == "__main__":
    main()
