#!/usr/bin/env python3
"""Migrate memory.db memories_vec from 768-dim (SPECTER2) to 1024-dim (bge-m3).

Drops and recreates memories_vec, then re-embeds every stored memory's content
with the new model. Safe to run repeatedly. The `memories` text rows are never
touched, so nothing is lost even if embedding fails.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import MEMORY_DB_PATH
from retrieval.embed import load_embed_model, embed_texts


def _open():
    import sqlite3, sqlite_vec
    c = sqlite3.connect(str(MEMORY_DB_PATH))
    c.enable_load_extension(True)
    sqlite_vec.load(c)
    c.enable_load_extension(False)
    c.row_factory = sqlite3.Row
    return c


def main():
    if not MEMORY_DB_PATH.exists():
        print("[mem] no memory.db — nothing to migrate")
        return
    c = _open()
    rows = c.execute("SELECT memory_id, content FROM memories").fetchall()
    print(f"[mem] {len(rows)} memories to re-embed")

    c.execute("DROP TABLE IF EXISTS memories_vec")
    c.execute(
        "CREATE VIRTUAL TABLE memories_vec USING vec0("
        "memory_id TEXT, embedding FLOAT[1024])"
    )
    c.commit()

    if rows:
        _, model = load_embed_model()
        ids = [r["memory_id"] for r in rows]
        texts = [r["content"] or "" for r in rows]
        vecs = embed_texts(None, model, texts)
        for mid, v in zip(ids, vecs):
            c.execute(
                "INSERT INTO memories_vec (memory_id, embedding) VALUES (?, ?)",
                (mid, np.asarray(v, dtype=np.float32).tobytes()),
            )
        c.commit()
    n = c.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]
    print(f"[mem] memories_vec rebuilt at 1024-dim: {n} rows")
    c.close()


if __name__ == "__main__":
    main()
