#!/usr/bin/env python3
"""
Stage 3 — Build the SQLite index (FTS5 + sqlite-vec).
Embeds chunks with SPECTER2 + proximity adapter on MPS.
Run: python3 -m ingest.build_index [--limit N]
"""
import argparse
import hashlib
import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DB_PATH, PARSED_DIR, EMBED_BATCH_SIZE, EMBED_DIM,
    EMBED_MODEL_BASE, EMBED_ADAPTER,
)
from ingest.chunk import chunk_paper
from retrieval.embed import load_embed_model, embed_texts
import re
from retrieval.graph import (
    init_graph_db,
    upsert_entity,
    upsert_relation,
    _open_db as _open_graph_db,
)

def _extract_dois(refs_text: str) -> list[str]:
    if not refs_text: return []
    matches = re.findall(r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+', refs_text)
    cleaned = []
    for m in matches:
        m = m.rstrip('.,;:)')
        if m not in cleaned:
            cleaned.append(m)
    return cleaned
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id  TEXT PRIMARY KEY,
    title     TEXT,
    authors   TEXT,
    year      INTEGER,
    source_pdf TEXT,
    doi       TEXT,            -- filled by optional metadata enrichment; NULL otherwise
    title_original TEXT        -- original PDF title before any cleanup; NULL otherwise
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    paper_id     TEXT REFERENCES papers(paper_id),
    section_name TEXT,
    page_start   INTEGER,
    page_end     INTEGER,
    position     INTEGER,
    text         TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    text,
    content='chunks',
    content_rowid='rowid'
);
"""

VEC_DDL = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
    chunk_id TEXT PRIMARY KEY,
    embedding FLOAT[{EMBED_DIM}]
);
"""


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    # sqlite-vec virtual table — must be loaded first
    conn.executescript(VEC_DDL)
    # Idempotent migration: add columns the app's queries expect (search/query
    # SELECT p.doi). Older indexes built before these columns existed get them
    # here so a re-index isn't required.
    have = {row[1] for row in conn.execute("PRAGMA table_info(papers)").fetchall()}
    for col in ("doi", "title_original"):
        if col not in have:
            conn.execute(f"ALTER TABLE papers ADD COLUMN {col} TEXT")
    conn.commit()


def _content_hash(parsed: dict) -> str:
    """Stable fingerprint of a paper's body text — for content-level dedup."""
    import re
    text = " ".join(s.get("text", "") for s in parsed.get("sections", []))
    norm = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return hashlib.md5(norm.encode()).hexdigest()


def _paper_exists(conn: sqlite3.Connection, paper_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,)
    ).fetchone()
    return row is not None


def _chunk_exists(conn: sqlite3.Connection, chunk_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM chunks WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    return row is not None


def _insert_paper(conn: sqlite3.Connection, parsed: dict) -> None:
    authors = parsed.get("authors", [])
    if isinstance(authors, list):
        authors = "; ".join(authors)
    conn.execute(
        "INSERT OR REPLACE INTO papers (paper_id, title, authors, year, source_pdf) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            parsed["paper_id"],
            parsed.get("title", ""),
            authors,
            parsed.get("year"),
            parsed.get("source_pdf", ""),
        ),
    )


def _insert_chunk(conn: sqlite3.Connection, chunk: dict) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO chunks "
        "(chunk_id, paper_id, section_name, page_start, page_end, position, text) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            chunk["chunk_id"],
            chunk["paper_id"],
            chunk["section_name"],
            chunk["page_start"],
            chunk["page_end"],
            chunk["position"],
            chunk["text"],
        ),
    )


def _insert_vec(conn: sqlite3.Connection, chunk_id: str, vec: np.ndarray) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, vec.astype(np.float32).tobytes()),
    )


def run(limit: int | None = None) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load sqlite-vec extension
    conn = sqlite3.connect(str(DB_PATH))
    conn.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    _init_db(conn)

    # Load SPECTER2
    log.info("Loading SPECTER2 + proximity adapter on MPS…")
    tokenizer, embed_model = load_embed_model()
    log.info("Model loaded.")

    init_graph_db()
    graph_conn = _open_graph_db()

    json_files = sorted(PARSED_DIR.glob("*.json"))
    if limit:
        json_files = json_files[:limit]

    seen_hashes: dict[str, str] = {}   # content_hash -> paper_id (first seen wins)
    skipped_dupes = 0

    for jf in tqdm(json_files, desc="Papers"):
        with open(jf, encoding="utf-8") as f:
            parsed = json.load(f)

        paper_id = parsed["paper_id"]

        # Content-level dedup guard: never index two papers with identical body
        # text (the recurring "same paper, different filename" problem).
        chash = _content_hash(parsed)
        if chash in seen_hashes:
            skipped_dupes += 1
            log.info("Skipping duplicate content: %s (== %s)", paper_id, seen_hashes[chash])
            continue
        seen_hashes[chash] = paper_id

        if not _paper_exists(conn, paper_id):
            _insert_paper(conn, parsed)
            conn.commit()
            
            title = parsed.get("title", paper_id)
            pid_entity = upsert_entity(graph_conn, name=title, entity_type="Paper", description=f"Paper ID: {paper_id}", paper_id=paper_id)
            refs_raw = parsed.get("references_raw", "")
            dois = _extract_dois(refs_raw)
            for doi in dois:
                doi_entity = upsert_entity(graph_conn, name=doi, entity_type="Paper", description=f"Cited DOI: {doi}", paper_id=paper_id)
                upsert_relation(graph_conn, pid_entity, doi_entity, relation="Cites", description="Reference", paper_id=paper_id)
            graph_conn.commit()

        chunks = chunk_paper(parsed)
        new_chunks = [c for c in chunks if not _chunk_exists(conn, c["chunk_id"])]

        if not new_chunks:
            continue

        # Embed in batches
        for i in range(0, len(new_chunks), EMBED_BATCH_SIZE):
            batch = new_chunks[i : i + EMBED_BATCH_SIZE]
            texts = [c["embedded_text"] for c in batch]
            titles = [parsed.get("title", "") for _ in batch]

            vecs = embed_texts(tokenizer, embed_model, texts, titles)

            for chunk, vec in zip(batch, vecs):
                _insert_chunk(conn, chunk)
                _insert_vec(conn, chunk["chunk_id"], vec)

            conn.commit()

    # Rebuild FTS5 index
    log.info("Rebuilding FTS5 index…")
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    conn.commit()

    total_papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    log.info("Index complete: %d papers, %d chunks. Skipped %d duplicate-content papers.",
             total_papers, total_chunks, skipped_dupes)
    conn.close()
    graph_conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Index only first N parsed JSON files (for testing)")
    args = parser.parse_args()
    run(args.limit)
