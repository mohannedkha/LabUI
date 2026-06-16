#!/usr/bin/env python3
"""
Stage 2 — Structure-aware chunking of parsed JSON papers.
Run: python3 -m ingest.chunk [--limit N]
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PARSED_DIR, CHUNK_TARGET_TOK, CHUNK_OVERLAP_TOK, CHARS_PER_TOKEN
)

CHUNK_TARGET_CHARS   = CHUNK_TARGET_TOK  * CHARS_PER_TOKEN   # ~3200
CHUNK_OVERLAP_CHARS  = CHUNK_OVERLAP_TOK * CHARS_PER_TOKEN   # ~600


def _chunk_id(paper_id: str, position: int) -> str:
    raw = f"{paper_id}::{position}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def chunk_section(
    paper_id: str,
    title: str,
    section: dict,
    start_position: int,
) -> list[dict]:
    text = section.get("text", "").strip()
    section_name = section.get("name", "Body")
    page_start = section.get("page_start", 1)
    page_end = section.get("page_end", page_start)

    if not text:
        return []

    # Short sections (Abstract, Conclusion, etc.) → single chunk
    if len(text) <= CHUNK_TARGET_CHARS:
        embedded_text = f"[{title} — {section_name}]\n{text}"
        return [
            {
                "chunk_id": _chunk_id(paper_id, start_position),
                "paper_id": paper_id,
                "section_name": section_name,
                "page_start": page_start,
                "page_end": page_end,
                "position": start_position,
                "text": text,
                "embedded_text": embedded_text,
            }
        ]

    chunks = []
    pos = 0
    chunk_index = start_position

    while pos < len(text):
        end = pos + CHUNK_TARGET_CHARS
        if end >= len(text):
            chunk_text = text[pos:]
        else:
            # Try to break at a sentence or paragraph boundary
            boundary = text.rfind("\n\n", pos, end)
            if boundary == -1 or boundary <= pos:
                boundary = text.rfind(". ", pos, end)
            if boundary == -1 or boundary <= pos:
                boundary = text.rfind("\n", pos, end)
            if boundary == -1 or boundary <= pos:
                boundary = end
            else:
                boundary += 1  # include the period/newline
            if text[boundary:boundary+1] == "\n":
                boundary += 1
            chunk_text = text[pos:boundary]

        chunk_text = chunk_text.strip()
        if chunk_text:
            embedded_text = f"[{title} — {section_name}]\n{chunk_text}"
            chunks.append(
                {
                    "chunk_id": _chunk_id(paper_id, chunk_index),
                    "paper_id": paper_id,
                    "section_name": section_name,
                    "page_start": page_start,
                    "page_end": page_end,
                    "position": chunk_index,
                    "text": chunk_text,
                    "embedded_text": embedded_text,
                }
            )
            chunk_index += 1

        # Advance with overlap
        next_pos = pos + CHUNK_TARGET_CHARS - CHUNK_OVERLAP_CHARS
        if next_pos <= pos:
            next_pos = pos + 1
        pos = next_pos

    return chunks


def chunk_paper(parsed: dict) -> list[dict]:
    paper_id = parsed["paper_id"]
    title = parsed.get("title", paper_id)
    all_chunks = []
    position = 0
    for section in parsed.get("sections", []):
        new_chunks = chunk_section(paper_id, title, section, position)
        all_chunks.extend(new_chunks)
        position += len(new_chunks)
    return all_chunks


def run(limit: int | None = None) -> list[dict]:
    json_files = sorted(PARSED_DIR.glob("*.json"))
    if limit:
        json_files = json_files[:limit]

    all_chunks = []
    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            parsed = json.load(f)
        chunks = chunk_paper(parsed)
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)} from {len(json_files)} papers")
    if all_chunks:
        sample = all_chunks[0]
        print(f"\nSample chunk:")
        print(f"  paper_id    : {sample['paper_id']}")
        print(f"  section     : {sample['section_name']}")
        print(f"  position    : {sample['position']}")
        print(f"  chars       : {len(sample['text'])}")
        print(f"  text[:200]  : {sample['text'][:200]}")

    return all_chunks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.limit)
