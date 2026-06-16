#!/usr/bin/env python3
"""
Remove duplicate papers that share a DOI, keeping the most complete copy
(most chunks). Cleans rag.db (papers, chunks, chunks_vec, chunks_fts) and the
graph (entity_chunks) so nothing dangles. Backs up both DBs first.

  python tools/dedup_by_doi.py            # dry run — show the plan, no writes
  python tools/dedup_by_doi.py --apply     # back up, remove, verify
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import sqlite_vec  # noqa: E402

GRAPH_DB = config.DB_PATH.parent / "graph.db"


def _open_vec(path):
    con = sqlite3.connect(str(path))
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    con.row_factory = sqlite3.Row
    return con


def plan(con):
    groups = con.execute(
        "SELECT doi FROM papers WHERE doi IS NOT NULL GROUP BY doi HAVING COUNT(*)>1"
    ).fetchall()
    drop_papers = []
    for g in groups:
        rows = con.execute(
            "SELECT p.paper_id, p.title, "
            "(SELECT COUNT(*) FROM chunks ch WHERE ch.paper_id=p.paper_id) nch "
            "FROM papers p WHERE p.doi=? ORDER BY nch DESC, p.paper_id",
            (g["doi"],),
        ).fetchall()
        keep, *drop = rows
        for d in drop:
            drop_papers.append((d["paper_id"], d["title"], d["nch"], keep["paper_id"], g["doi"]))
    return drop_papers


def main(apply: bool):
    con = _open_vec(config.DB_PATH)
    drop = plan(con)
    print(f"Duplicate groups -> {len(drop)} papers to remove (keeping the copy with most chunks):\n")
    for pid, title, nch, keep, doi in drop:
        print(f"  DROP {pid}  (chunks={nch})  [{doi}]")
        print(f"       title: {title[:64]!r}  -> keep {keep}")
    if not drop:
        print("Nothing to do."); return
    if not apply:
        print("\n(dry run — pass --apply to perform removal)"); con.close(); return

    drop_ids = [d[0] for d in drop]
    ph = ",".join("?" * len(drop_ids))
    chunk_ids = [r[0] for r in con.execute(
        f"SELECT chunk_id FROM chunks WHERE paper_id IN ({ph})", drop_ids)]

    # Backups
    for p in (config.DB_PATH, GRAPH_DB):
        b = p.with_suffix(p.suffix + f".pre_dedup_{datetime.now():%Y%m%d_%H%M%S}")
        shutil.copy2(p, b)
        print(f"\nBacked up {p.name} -> {b.name}")

    # rag.db deletions
    cph = ",".join("?" * len(chunk_ids))
    if chunk_ids:
        con.execute(f"DELETE FROM chunks_vec WHERE chunk_id IN ({cph})", chunk_ids)
    con.execute(f"DELETE FROM chunks WHERE paper_id IN ({ph})", drop_ids)
    con.execute(f"DELETE FROM papers WHERE paper_id IN ({ph})", drop_ids)
    con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    con.commit()

    # graph cleanup
    g = sqlite3.connect(str(GRAPH_DB))
    if chunk_ids:
        g.execute(f"DELETE FROM entity_chunks WHERE chunk_id IN ({cph})", chunk_ids)
    g.commit()

    # Verify
    print("\n=== verification ===")
    print("papers:", con.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
    print("chunks:", con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    print("vectors:", con.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0])
    print("fts:", con.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0])
    print("chunks w/o vector:", con.execute(
        "SELECT COUNT(*) FROM chunks c LEFT JOIN chunks_vec v ON v.chunk_id=c.chunk_id "
        "WHERE v.chunk_id IS NULL").fetchone()[0])
    print("duplicate DOIs remaining:", con.execute(
        "SELECT COUNT(*) FROM (SELECT doi FROM papers WHERE doi IS NOT NULL "
        "GROUP BY doi HAVING COUNT(*)>1)").fetchone()[0])
    dangling = con.execute(
        f"SELECT COUNT(*) FROM ({'SELECT chunk_id FROM chunks'})").fetchone()[0]
    gd = g.execute("SELECT COUNT(*) FROM entity_chunks").fetchone()[0]
    print("entity_chunks rows:", gd)
    g.close(); con.close()
    print("\nDone.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true")
    main(ap.parse_args().apply)
