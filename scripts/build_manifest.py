"""
Build corpus manifest from data/raw PDF files.

Scans data/raw recursively, collects metadata for each PDF,
and writes an Excel manifest to data/raw/corpus_manifest.xlsx.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def build_manifest(raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Scan *raw_dir* recursively and build a manifest DataFrame."""
    raw_path = Path(raw_dir).resolve()
    records: list[dict[str, str]] = []

    for pdf_path in raw_path.rglob("*.pdf"):
        # Compute relative parts from raw_dir root
        rel_parts = pdf_path.relative_to(raw_path).parts
        if len(rel_parts) < 2:
            # PDF sits directly under data/raw — skip or handle gracefully
            continue

        brand = rel_parts[0]
        category = rel_parts[1]
        file_name = pdf_path.name
        local_path = str(pdf_path.resolve())

        records.append(
            {
                "brand": brand,
                "category": category,
                "file_name": file_name,
                "local_path": local_path,
                "source_url": "",
                "model": "",
                "doc_type": "user_manual",
            }
        )

    df = pd.DataFrame(
        records,
        columns=[
            "brand",
            "category",
            "file_name",
            "local_path",
            "source_url",
            "model",
            "doc_type",
        ],
    )
    return df


def main() -> None:
    raw_dir = Path("data/raw")
    output_path = raw_dir / "corpus_manifest.xlsx"

    df = build_manifest(raw_dir)

    if df.empty:
        print("No PDF files found under data/raw.")
        return

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"Manifest written to {output_path} ({len(df)} entries)")

    # Print category-level counts
    counts = df.groupby("category").size().sort_index()
    print("\nCategory PDF counts:")
    for category, count in counts.items():
        print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
