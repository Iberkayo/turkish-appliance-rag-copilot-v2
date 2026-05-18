"""
Contextual chunking pipeline.

Reads page-level JSONL, groups by document, and produces overlapping chunks
with metadata-prepended contextual text suitable for downstream embedding.

Outputs:
- data/processed/chunks.jsonl
- data/eval/chunk_audit.xlsx
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
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
PAGES_JSONL = Path("data/processed/pages.jsonl")
CHUNKS_JSONL = Path("data/processed/chunks.jsonl")
CHUNK_AUDIT_PATH = Path("data/eval/chunk_audit.xlsx")

MAX_WORDS = 900
MIN_WORDS = 120
OVERLAP_WORDS = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_pages(path: Path) -> list[dict[str, Any]]:
    """Load all page records from a JSONL file."""
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d page records from %s", len(records), path)
    return records


def get_overlap(text: str, overlap_words: int) -> str:
    """Return the last *overlap_words* words from *text*."""
    words = text.split()
    if len(words) <= overlap_words:
        return text
    return " ".join(words[-overlap_words:])


def split_long_text(text: str, max_words: int, overlap_words: int) -> list[str]:
    """Split a long text into sentence-respecting chunks with overlap."""
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current_sents: list[str] = []
    current_words = 0

    for sent in sentences:
        sent_words = len(sent.split())
        if current_words + sent_words <= max_words:
            current_sents.append(sent)
            current_words += sent_words
        else:
            if current_sents:
                chunk_text = " ".join(current_sents)
                chunks.append(chunk_text)
                overlap = get_overlap(chunk_text, overlap_words)
                current_sents = ([overlap] if overlap else []) + [sent]
                current_words = len(overlap.split()) + sent_words
            else:
                # Single sentence exceeds max_words — word-level split fallback
                words = sent.split()
                step = max_words - overlap_words
                for i in range(0, len(words), step):
                    slice_words = words[i : i + max_words]
                    chunks.append(" ".join(slice_words))
                current_sents = []
                current_words = 0

    if current_sents:
        chunks.append(" ".join(current_sents))

    return chunks


def build_contextual_text(
    meta: dict[str, str], page_start: int, page_end: int, text: str
) -> str:
    """Prepend metadata header to chunk text."""
    pages_str = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
    header = (
        f"Brand: {meta['brand']}\n"
        f"Category: {meta['category']}\n"
        f"Document: {meta['file_name']}\n"
        f"Pages: {pages_str}\n\n"
    )
    return header + text


def chunk_document(pages: list[dict[str, Any]], meta: dict[str, str]) -> list[dict[str, Any]]:
    """Create overlapping chunks from a single document's pages."""
    chunks: list[dict[str, Any]] = []
    buffer_pages: list[int] = []
    buffer_texts: list[str] = []
    buffer_word_count = 0
    overlap_text = ""
    chunk_index = 0

    def flush() -> None:
        nonlocal buffer_pages, buffer_texts, buffer_word_count, overlap_text, chunk_index
        if not buffer_pages:
            return

        full_text = overlap_text
        if full_text and buffer_texts:
            full_text += "\n\n"
        full_text += "\n\n".join(buffer_texts)

        chunk_index += 1
        chunk_id = f"{meta['doc_id']}_chunk_{chunk_index:04d}"
        contextual = build_contextual_text(meta, buffer_pages[0], buffer_pages[-1], full_text)

        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": meta["doc_id"],
                "brand": meta["brand"],
                "category": meta["category"],
                "file_name": meta["file_name"],
                "page_start": buffer_pages[0],
                "page_end": buffer_pages[-1],
                "chunk_index": chunk_index,
                "text": full_text,
                "contextual_text": contextual,
                "char_count": len(full_text),
                "word_count": len(full_text.split()),
            }
        )

        overlap_text = get_overlap(full_text, OVERLAP_WORDS)
        buffer_pages = []
        buffer_texts = []
        buffer_word_count = 0

    for page in pages:
        text = page.get("text", "").strip()
        page_num = page["page_number"]
        word_count = len(text.split()) if text else 0

        if not text:
            continue

        # Single page too long → split independently
        if word_count > MAX_WORDS:
            flush()
            overlap_text = ""

            sub_chunks = split_long_text(text, MAX_WORDS, OVERLAP_WORDS)
            for sub_text in sub_chunks:
                chunk_index += 1
                chunk_id = f"{meta['doc_id']}_chunk_{chunk_index:04d}"
                contextual = build_contextual_text(meta, page_num, page_num, sub_text)

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": meta["doc_id"],
                        "brand": meta["brand"],
                        "category": meta["category"],
                        "file_name": meta["file_name"],
                        "page_start": page_num,
                        "page_end": page_num,
                        "chunk_index": chunk_index,
                        "text": sub_text,
                        "contextual_text": contextual,
                        "char_count": len(sub_text),
                        "word_count": len(sub_text.split()),
                    }
                )
                overlap_text = get_overlap(sub_text, OVERLAP_WORDS)

            continue

        # Greedy add to buffer (account for upcoming overlap text)
        overlap_buffer_words = len(overlap_text.split()) if overlap_text else 0
        if buffer_word_count + overlap_buffer_words + word_count <= MAX_WORDS:
            buffer_pages.append(page_num)
            buffer_texts.append(text)
            buffer_word_count += word_count
        else:
            flush()
            buffer_pages = [page_num]
            buffer_texts = [text]
            buffer_word_count = word_count

    flush()
    return chunks


def run_quality_checks(chunks: list[dict[str, Any]]) -> None:
    """Run lightweight sanity checks on the generated chunks."""
    logger.info("Running quality checks ...")

    # 1. Unique chunk_ids
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk_id detected!"

    # 2. No empty text
    empty = [c["chunk_id"] for c in chunks if not c["text"].strip()]
    assert not empty, f"Empty text chunks found: {empty}"

    # 3. Each chunk references exactly one doc_id (implicit because we chunk per doc)
    # We simply verify the doc_id prefix matches.
    for c in chunks:
        expected_prefix = c["doc_id"]
        assert c["chunk_id"].startswith(expected_prefix), (
            f"chunk_id {c['chunk_id']} does not match doc_id {expected_prefix}"
        )

    # 4. contextual_text is longer than raw text
    short_ctx = [
        c["chunk_id"]
        for c in chunks
        if len(c["contextual_text"]) <= len(c["text"])
    ]
    assert not short_ctx, f"contextual_text not longer than text for: {short_ctx}"

    logger.info("All quality checks passed.")


def write_jsonl(chunks: list[dict[str, Any]], path: Path) -> None:
    """Persist chunks to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    logger.info("Written %d chunks to %s", len(chunks), path)


def build_audit_df(chunks: list[dict[str, Any]]) -> pd.DataFrame:
    """Build an audit DataFrame from chunk records."""
    df = pd.DataFrame(chunks)
    df["text_preview"] = df["text"].str[:200]
    audit_cols = [
        "chunk_id",
        "doc_id",
        "brand",
        "category",
        "file_name",
        "page_start",
        "page_end",
        "chunk_index",
        "word_count",
        "char_count",
        "text_preview",
    ]
    return df[audit_cols]


def print_summary(chunks: list[dict[str, Any]]) -> None:
    """Print terminal summary."""
    total_chunks = len(chunks)
    word_counts = [c["word_count"] for c in chunks]
    total_words = sum(word_counts)
    avg_words = total_words / total_chunks if total_chunks else 0
    min_words = min(word_counts) if word_counts else 0
    max_words = max(word_counts) if word_counts else 0
    short_chunks = sum(1 for w in word_counts if w < MIN_WORDS)
    long_chunks = sum(1 for w in word_counts if w > MAX_WORDS)

    df = pd.DataFrame(chunks)
    cat_counts = df.groupby("category").size().sort_index()

    print("\n" + "=" * 60)
    print("CHUNK BUILDER SUMMARY")
    print("=" * 60)
    print(f"Total chunks       : {total_chunks}")
    print(f"Total words        : {total_words:,}")
    print(f"Avg words/chunk    : {avg_words:.0f}")
    print(f"Min words/chunk    : {min_words}")
    print(f"Max words/chunk    : {max_words}")
    print(f"Chunks < {MIN_WORDS} w   : {short_chunks}")
    print(f"Chunks > {MAX_WORDS} w   : {long_chunks}")
    print("=" * 60)
    print("\n--- Category chunk counts ---")
    for category, count in cat_counts.items():
        print(f"  {category}: {count}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    pages = load_pages(PAGES_JSONL)

    # Group by doc_id and sort by page_number within each group
    docs: dict[str, list[dict[str, Any]]] = {}
    for page in pages:
        docs.setdefault(page["doc_id"], []).append(page)

    for doc_id in docs:
        docs[doc_id].sort(key=lambda p: p["page_number"])

    all_chunks: list[dict[str, Any]] = []

    for doc_id, doc_pages in docs.items():
        meta = {
            "doc_id": doc_id,
            "brand": doc_pages[0]["brand"],
            "category": doc_pages[0]["category"],
            "file_name": doc_pages[0]["file_name"],
        }
        chunks = chunk_document(doc_pages, meta)
        all_chunks.extend(chunks)
        logger.info("%s → %d chunks", doc_id, len(chunks))

    run_quality_checks(all_chunks)

    write_jsonl(all_chunks, CHUNKS_JSONL)

    audit_df = build_audit_df(all_chunks)
    CHUNK_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_excel(CHUNK_AUDIT_PATH, index=False)
    logger.info("Chunk audit saved to %s", CHUNK_AUDIT_PATH)

    print_summary(all_chunks)


if __name__ == "__main__":
    main()
