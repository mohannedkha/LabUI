#!/usr/bin/env python3
"""
Repair papers that were indexed with garbled text (broken PDF font encoding).

Such papers extract as glyph soup (raw glyph names like "/a114" and decorative
Unicode standing in for letters). parse_pdfs now re-OCRs these automatically on
first parse, but papers already in the index keep their cached garbled JSON and
their stale chunks/vectors. This tool finds them, purges them, and rebuilds them
cleanly through the OCR-aware parser.

Run:  python3 -m ingest.reprocess_garbled [--dry-run] [--threshold 0.25]

It is safe to run repeatedly: clean papers are left untouched.
"""
import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH, PARSED_DIR
from ingest.parse_pdfs import is_garbled, garbage_ratio, _sections_text
from ingest import parse_pdfs, build_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def find_garbled(threshold: float = 0.25) -> list[tuple[str, Path, float]]:
    """Return (paper_id, json_path, ratio) for every garbled parsed paper."""
    out: list[tuple[str, Path, float]] = []
    for jf in sorted(PARSED_DIR.glob("*.json")):
        try:
            parsed = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Could not read %s: %s", jf.name, e)
            continue
        sections = parsed.get("sections", [])
        title = parsed.get("title")
        if is_garbled(sections, title, threshold=threshold):
            ratio = garbage_ratio(_sections_text(sections, title))
            out.append((parsed.get("paper_id", jf.stem), jf, ratio))
    return out


def _open_index_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def purge_paper(conn: sqlite3.Connection, paper_id: str) -> int:
    """Delete a paper's chunks, vectors, and row. Returns chunks removed."""
    chunk_ids = [r[0] for r in conn.execute(
        "SELECT chunk_id FROM chunks WHERE paper_id = ?", (paper_id,)
    ).fetchall()]
    for cid in chunk_ids:
        conn.execute("DELETE FROM chunks_vec WHERE chunk_id = ?", (cid,))
    conn.execute("DELETE FROM chunks WHERE paper_id = ?", (paper_id,))
    conn.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,))
    conn.commit()
    return len(chunk_ids)


def run(dry_run: bool = False, threshold: float = 0.25) -> dict:
    garbled = find_garbled(threshold)
    if not garbled:
        log.info("No garbled papers found — nothing to do.")
        return {"found": 0, "repaired": 0, "papers": []}

    log.info("Found %d garbled paper(s):", len(garbled))
    for pid, _jf, ratio in garbled:
        log.info("  %-50s garbage=%.0f%%", pid[:50], ratio * 100)

    if dry_run:
        return {"found": len(garbled), "repaired": 0,
                "papers": [pid for pid, _, _ in garbled]}

    # 1) Purge stale index rows + cached parses so they get rebuilt from scratch.
    conn = _open_index_db() if DB_PATH.exists() else None
    try:
        for pid, jf, _ratio in garbled:
            if conn is not None:
                removed = purge_paper(conn, pid)
                log.info("Purged %s (%d chunks)", pid[:50], removed)
            try:
                jf.unlink()
            except FileNotFoundError:
                pass
    finally:
        if conn is not None:
            conn.close()

    # 2) Re-parse (OCR-aware) and re-index the now-missing papers.
    log.info("Re-parsing with OCR fallback…")
    parse_pdfs.run()
    log.info("Re-indexing…")
    build_index.run()

    return {"found": len(garbled), "repaired": len(garbled),
            "papers": [pid for pid, _, _ in garbled]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="List garbled papers without changing anything")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Garbage ratio above which a paper is re-OCR'd (0..1)")
    args = parser.parse_args()
    result = run(dry_run=args.dry_run, threshold=args.threshold)
    print(json.dumps(result, indent=2))
