"""
Hybrid retrieval: BM25 (FTS5) + dense (SPECTER2) + RRF + rerank + diversify.
"""
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DB_PATH,
    BM25_TOP_N, DENSE_TOP_N, RRF_K, RERANK_TOP_N,
    FINAL_TOP_K, MAX_PER_PAPER, MIN_DIVERSE,
)


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    return conn


def _sanitize_fts(query: str) -> str:
    """Escape FTS5 special characters."""
    # Remove chars that break FTS5 queries
    query = re.sub(r'["\(\)\*\:\^]', ' ', query)
    query = query.strip()
    return f'"{query}"' if query else '""'


def _bm25_search(conn: sqlite3.Connection, query: str, top_n: int, selected_paper_ids: list[str] = None) -> list[str]:
    """Return chunk_ids ranked by BM25."""
    fts_query = _sanitize_fts(query)
    filter_clause = ""
    params = [fts_query]
    
    if selected_paper_ids:
        placeholders = ",".join("?" * len(selected_paper_ids))
        filter_clause = f" AND c.paper_id IN ({placeholders}) "
        params.extend(selected_paper_ids)
    
    params.append(top_n)

    try:
        rows = conn.execute(
            f"""
            SELECT c.chunk_id
            FROM chunks_fts f
            JOIN chunks c ON c.chunk_id = f.chunk_id
            WHERE chunks_fts MATCH ? {filter_clause}
            ORDER BY rank
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [r["chunk_id"] for r in rows]
    except sqlite3.OperationalError:
        # Fallback: try unquoted query
        plain = re.sub(r'[^\w\s]', ' ', query).strip()
        if not plain:
            return []
        try:
            plain_params = [plain]
            if selected_paper_ids:
                plain_params.extend(selected_paper_ids)
            plain_params.append(top_n)
            
            rows = conn.execute(
                f"""
                SELECT c.chunk_id
                FROM chunks_fts f
                JOIN chunks c ON c.chunk_id = f.chunk_id
                WHERE chunks_fts MATCH ? {filter_clause}
                ORDER BY rank
                LIMIT ?
                """,
                tuple(plain_params),
            ).fetchall()
            return [r["chunk_id"] for r in rows]
        except Exception:
            return []


def _dense_search(
    conn: sqlite3.Connection, query_vec: np.ndarray, top_n: int, selected_paper_ids: list[str] = None
) -> list[str]:
    """Return chunk_ids ranked by cosine similarity via sqlite-vec."""
    vec_bytes = query_vec.astype(np.float32).tobytes()
    filter_clause = ""
    params = [vec_bytes, top_n]
    
    if selected_paper_ids:
        placeholders = ",".join("?" * len(selected_paper_ids))
        filter_clause = f" AND c.paper_id IN ({placeholders}) "
        params.extend(selected_paper_ids)
        
    rows = conn.execute(
        f"""
        SELECT v.chunk_id
        FROM chunks_vec v
        JOIN chunks c ON c.chunk_id = v.chunk_id
        WHERE v.embedding MATCH ?
          AND v.k = ?
          {filter_clause}
        ORDER BY v.distance
        """,
        tuple(params),
    ).fetchall()
    return [r["chunk_id"] for r in rows]


def _rrf(lists: list[list[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal Rank Fusion over multiple ranked lists."""
    scores: dict[str, float] = {}
    for ranked in lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda x: -scores[x])


def _fetch_chunks(conn: sqlite3.Connection, chunk_ids: list[str]) -> list[dict]:
    if not chunk_ids:
        return []
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"""
        SELECT c.chunk_id, c.paper_id, c.section_name, c.page_start, c.page_end, c.text,
               p.title, p.authors, p.year, p.doi
        FROM chunks c
        JOIN papers p ON p.paper_id = c.paper_id
        WHERE c.chunk_id IN ({placeholders})
        """,
        chunk_ids,
    ).fetchall()
    # Preserve order from chunk_ids list
    by_id = {r["chunk_id"]: dict(r) for r in rows}
    return [by_id[cid] for cid in chunk_ids if cid in by_id]


def _diversify(chunks: list[dict], max_per_paper: int, min_total: int) -> list[dict]:
    """Cap results per paper to avoid one paper dominating."""
    counts: dict[str, int] = {}
    primary, overflow = [], []
    for c in chunks:
        pid = c["paper_id"]
        if counts.get(pid, 0) < max_per_paper:
            primary.append(c)
            counts[pid] = counts.get(pid, 0) + 1
        else:
            overflow.append(c)

    if len(primary) < min_total:
        needed = min_total - len(primary)
        primary.extend(overflow[:needed])

    return primary


def search(
    query: str,
    query_vec: np.ndarray,
    reranker,
    top_k: int = FINAL_TOP_K,
    selected_paper_ids: list[str] = None,
) -> list[dict]:
    """
    Full hybrid retrieval pipeline.
    query_vec: shape (1024,) float32 numpy array from bge-m3.
    reranker: loaded CrossEncoder from retrieval.rerank.
    Returns list of chunk dicts with metadata.
    """
    from retrieval.rerank import rerank as do_rerank

    conn = _open_db()

    bm25_ids  = _bm25_search(conn, query, BM25_TOP_N, selected_paper_ids)
    dense_ids = _dense_search(conn, query_vec, DENSE_TOP_N, selected_paper_ids)

    fused_ids = _rrf([bm25_ids, dense_ids])[:RERANK_TOP_N]
    candidates = _fetch_chunks(conn, fused_ids)

    conn.close()

    if not candidates:
        return []

    reranked = do_rerank(reranker, query, candidates, top_n=top_k * 2)
    diversified = _diversify(reranked, MAX_PER_PAPER, MIN_DIVERSE)
    return diversified[:top_k]
