"""
Persistent chat history and semantic agent memory.

Schema (memory.db):
  sessions   — one row per chat session
  turns      — user / assistant messages within a session
  memories   — extracted research findings (semantic + FTS searchable)
  memories_vec (vec0) — bge-m3 (1024-dim) embeddings of memories
  memories_fts (fts5) — keyword index of memory content
"""
import json
import re
import sqlite3
import time
import uuid
from pathlib import Path

import numpy as np

from config import MEMORY_DB_PATH


# ── DB connection ─────────────────────────────────────────────────────────────

def _open_db() -> sqlite3.Connection:
    MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(MEMORY_DB_PATH))
    conn.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


# ── Schema init ───────────────────────────────────────────────────────────────

def init_db() -> None:
    conn = _open_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id  TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            agent_id    TEXT NOT NULL DEFAULT 'chat',
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL,
            turn_count  INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS turns (
            turn_id     TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL
                            REFERENCES sessions(session_id) ON DELETE CASCADE,
            turn_index  INTEGER NOT NULL,
            role        TEXT NOT NULL,          -- 'user' | 'assistant'
            content     TEXT NOT NULL,
            agent_id    TEXT,
            papers_json TEXT DEFAULT '[]',      -- JSON list of paper_ids cited
            sources_json TEXT DEFAULT '[]',     -- full source cards (n, title, doi, pages…)
            web_json    TEXT DEFAULT '[]',      -- web results for this turn
            created_at  REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_turns_session
            ON turns(session_id, turn_index);

        CREATE TABLE IF NOT EXISTS memories (
            memory_id       TEXT PRIMARY KEY,
            content         TEXT NOT NULL,
            source_session  TEXT,
            source_query    TEXT,
            memory_type     TEXT NOT NULL DEFAULT 'finding',
            importance      REAL NOT NULL DEFAULT 0.5,
            created_at      REAL NOT NULL,
            access_count    INTEGER NOT NULL DEFAULT 0,
            last_accessed   REAL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content,
            memory_id UNINDEXED
        );
    """)

    # Idempotent migration for DBs created before per-turn sources/web existed.
    have = {r[1] for r in conn.execute("PRAGMA table_info(turns)").fetchall()}
    for col in ("sources_json", "web_json"):
        if col not in have:
            conn.execute(f"ALTER TABLE turns ADD COLUMN {col} TEXT DEFAULT '[]'")

    # memories_vec — requires sqlite-vec (graceful no-op if unavailable)
    try:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                memory_id TEXT,
                embedding FLOAT[1024]
            )
        """)
    except Exception:
        pass

    conn.commit()
    conn.close()


# ── Sessions ──────────────────────────────────────────────────────────────────

def create_session(title: str, agent_id: str = "chat") -> str:
    session_id = str(uuid.uuid4())
    now = time.time()
    conn = _open_db()
    conn.execute(
        "INSERT INTO sessions (session_id, title, agent_id, created_at, updated_at)"
        " VALUES (?,?,?,?,?)",
        (session_id, title[:120], agent_id, now, now),
    )
    conn.commit()
    conn.close()
    return session_id


def get_sessions(limit: int = 80) -> list[dict]:
    conn = _open_db()
    rows = conn.execute(
        "SELECT session_id, title, agent_id, created_at, updated_at, turn_count"
        " FROM sessions ORDER BY updated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> dict | None:
    conn = _open_db()
    row = conn.execute(
        "SELECT session_id, title, agent_id, created_at, updated_at, turn_count"
        " FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if not row:
        conn.close()
        return None

    turns = conn.execute(
        "SELECT turn_id, turn_index, role, content, agent_id, papers_json,"
        " sources_json, web_json, created_at"
        " FROM turns WHERE session_id = ? ORDER BY turn_index",
        (session_id,),
    ).fetchall()
    conn.close()

    return {
        **dict(row),
        "turns": [
            {
                **dict(t),
                "papers": json.loads(t["papers_json"] or "[]"),
                "sources": json.loads(t["sources_json"] or "[]"),
                "web": json.loads(t["web_json"] or "[]"),
            }
            for t in turns
        ],
    }


def add_turn(
    session_id: str,
    role: str,
    content: str,
    agent_id: str | None = None,
    papers: list[dict] | None = None,
    sources: list[dict] | None = None,
    web: list[dict] | None = None,
) -> str:
    turn_id = str(uuid.uuid4())
    now = time.time()
    conn = _open_db()

    idx = conn.execute(
        "SELECT COALESCE(MAX(turn_index)+1, 0) FROM turns WHERE session_id=?",
        (session_id,),
    ).fetchone()[0]

    # `sources` are the full numbered source cards shown in the UI; `papers` is a
    # legacy id-only list. Persist the full cards + web results so a reopened
    # conversation keeps its sources.
    cards = sources if sources is not None else (papers or [])
    papers_json = json.dumps([p.get("paper_id", "") for p in cards])
    sources_json = json.dumps(sources or [])
    web_json = json.dumps(web or [])

    conn.execute(
        "INSERT INTO turns"
        " (turn_id, session_id, turn_index, role, content, agent_id,"
        "  papers_json, sources_json, web_json, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (turn_id, session_id, idx, role, content, agent_id,
         papers_json, sources_json, web_json, now),
    )
    conn.execute(
        "UPDATE sessions SET updated_at=?, turn_count=turn_count+1"
        " WHERE session_id=?",
        (now, session_id),
    )
    conn.commit()
    conn.close()
    return turn_id


def delete_session(session_id: str) -> None:
    conn = _open_db()
    conn.execute("DELETE FROM turns WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
    conn.commit()
    conn.close()


def rename_session(session_id: str, title: str) -> None:
    conn = _open_db()
    conn.execute(
        "UPDATE sessions SET title=?, updated_at=? WHERE session_id=?",
        (title[:120], time.time(), session_id),
    )
    conn.commit()
    conn.close()


# ── Memory ────────────────────────────────────────────────────────────────────

def save_memory(
    content: str,
    embedding: np.ndarray,
    session_id: str | None = None,
    source_query: str | None = None,
    memory_type: str = "finding",
    importance: float = 0.5,
) -> str:
    memory_id = str(uuid.uuid4())
    now = time.time()
    conn = _open_db()

    conn.execute(
        "INSERT INTO memories"
        " (memory_id, content, source_session, source_query, memory_type, importance, created_at)"
        " VALUES (?,?,?,?,?,?,?)",
        (memory_id, content, session_id, source_query, memory_type, importance, now),
    )
    conn.execute(
        "INSERT INTO memories_fts (content, memory_id) VALUES (?,?)",
        (content, memory_id),
    )
    try:
        vec_bytes = embedding.astype(np.float32).tobytes()
        conn.execute(
            "INSERT INTO memories_vec (memory_id, embedding) VALUES (?,?)",
            (memory_id, vec_bytes),
        )
    except Exception:
        pass

    conn.commit()
    conn.close()
    return memory_id


def search_memories(
    query_vec: np.ndarray,
    query_text: str,
    top_k: int = 5,
) -> list[dict]:
    """Hybrid memory retrieval: vector kNN + FTS5, RRF-merged."""
    conn = _open_db()
    scores: dict[str, dict] = {}

    # Vector search (same MATCH/k syntax as chunks_vec)
    try:
        vec_bytes = query_vec.astype(np.float32).tobytes()
        rows = conn.execute(
            "SELECT m.memory_id, m.content, m.memory_type, m.importance,"
            "       m.source_query, m.created_at, v.distance"
            " FROM memories_vec v"
            " JOIN memories m ON m.memory_id = v.memory_id"
            " WHERE v.embedding MATCH ? AND k = ?"
            " ORDER BY v.distance",
            (vec_bytes, top_k * 3),
        ).fetchall()
        for rank, r in enumerate(rows, 1):
            scores[r["memory_id"]] = {
                **dict(r), "rrf": 1.0 / (60 + rank)
            }
    except Exception:
        pass

    # FTS search
    try:
        safe = re.sub(r'["\(\)\*\:\^]', " ", query_text).strip()
        if safe:
            fts_rows = conn.execute(
                "SELECT memory_id, content FROM memories_fts"
                " WHERE memories_fts MATCH ? LIMIT ?",
                (f'"{safe}"', top_k * 3),
            ).fetchall()
            for rank, r in enumerate(fts_rows, 1):
                mid = r["memory_id"]
                rrf_score = 1.0 / (60 + rank)
                if mid in scores:
                    scores[mid]["rrf"] += rrf_score
                else:
                    meta = conn.execute(
                        "SELECT memory_id, content, memory_type, importance,"
                        "       source_query, created_at"
                        " FROM memories WHERE memory_id=?",
                        (mid,),
                    ).fetchone()
                    if meta:
                        scores[mid] = {**dict(meta), "rrf": rrf_score}
    except Exception:
        pass

    # Update access stats for returned memories
    if scores:
        now = time.time()
        placeholders = ",".join("?" * len(scores))
        conn.execute(
            f"UPDATE memories SET access_count=access_count+1, last_accessed=?"
            f" WHERE memory_id IN ({placeholders})",
            [now] + list(scores.keys()),
        )
        conn.commit()

    conn.close()

    ranked = sorted(scores.values(), key=lambda x: -x["rrf"])
    return ranked[:top_k]


def get_all_memories(limit: int = 200) -> list[dict]:
    conn = _open_db()
    rows = conn.execute(
        "SELECT memory_id, content, memory_type, importance,"
        "       source_query, created_at, access_count"
        " FROM memories ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_memory(memory_id: str) -> None:
    conn = _open_db()
    conn.execute("DELETE FROM memories WHERE memory_id=?", (memory_id,))
    conn.execute("DELETE FROM memories_fts WHERE memory_id=?", (memory_id,))
    try:
        conn.execute("DELETE FROM memories_vec WHERE memory_id=?", (memory_id,))
    except Exception:
        pass
    conn.commit()
    conn.close()


# ── Memory extraction (called after each LLM response) ───────────────────────

_EXTRACT_PROMPT = """\
You are extracting research memory from a Q&A exchange.

Question: {query}

Answer (excerpt): {answer}

Extract 1-3 concise factual findings worth remembering for future research queries.
Rules:
- Each finding must be a single sentence stating a specific scientific fact, \
measurement, relationship, or mechanism.
- Only include findings explicitly stated in the answer above.
- Format: one finding per line, each starting with "- ".
- If there are no notable findings worth remembering, respond exactly: none
"""


def extract_and_save_memories(
    query: str,
    answer: str,
    session_id: str | None,
    tokenizer,
    embed_model,
    ollama_url: str,
    gen_model: str,
) -> list[str]:
    """
    Ask the LLM to extract key findings from a Q&A exchange,
    embed them with SPECTER2, and store in memory.
    Returns list of saved memory_ids (may be empty).
    """
    import requests as _req

    # Trim answer to ~1200 chars to keep extraction prompt short
    answer_excerpt = answer[:1200] + ("…" if len(answer) > 1200 else "")

    prompt = _EXTRACT_PROMPT.format(
        query=query[:300],
        answer=answer_excerpt,
    )

    try:
        resp = _req.post(
            f"{ollama_url}/v1/chat/completions",
            json={
                "model": gen_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        text = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"[memory] extraction request failed: {e}")
        return []

    # Reasoning models (Qwen3 / R1 / QwQ) prepend a <think>…</think> block — drop it
    # so chain-of-thought lines aren't mistaken for extracted findings.
    import re as _re
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL | _re.IGNORECASE).strip()
    if "</think>" in text and "<think>" not in text:
        text = text.split("</think>", 1)[1].strip()

    if not text or text.lower().startswith("none"):
        return []

    # Parse bullet lines
    findings = []
    for line in text.split("\n"):
        line = line.strip().lstrip("-•*").strip()
        if len(line) > 20:          # skip empty / trivial lines
            findings.append(line)
    if not findings:
        return []

    # Embed and store each finding
    from retrieval.embed import embed_texts as _embed_texts
    saved = []
    try:
        vecs = _embed_texts(tokenizer, embed_model, findings, [""] * len(findings))
        for finding, vec in zip(findings, vecs):
            mid = save_memory(
                content=finding,
                embedding=vec,
                session_id=session_id,
                source_query=query[:200],
                memory_type="finding",
                importance=0.6,
            )
            saved.append(mid)
        print(f"[memory] saved {len(saved)} finding(s) from session {session_id}")
    except Exception as e:
        print(f"[memory] embedding/storage failed: {e}")

    return saved
