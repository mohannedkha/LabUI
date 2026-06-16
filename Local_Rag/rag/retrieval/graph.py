"""
Graph-RAG: entity/relation storage and retrieval.

Schema (graph.db, separate from rag.db):
  entities       — named entities extracted from chunks
  relations      — directional relations between entities
  entity_chunks  — many-to-many: entity ↔ chunk_id
  communities    — LLM-summarised entity clusters (optional)
  entities_fts   — FTS5 on entity name + description

Query flow:
  1. FTS search entity names for query terms
  2. Expand to 1-hop neighbours via relations
  3. Collect chunk_ids linked to all found entities
  4. Return as supplementary retrieval results
"""
import json
import re
import sqlite3
import time
import uuid
from pathlib import Path

from config import DATA_DIR


GRAPH_DB_PATH = DATA_DIR / "graph.db"

# Entity types + display colours (used by UI)
ENTITY_COLORS = {
    "Chemical":    "#10b981",   # emerald
    "Method":      "#3b82f6",   # blue
    "Concept":     "#8b5cf6",   # violet
    "Measurement": "#f59e0b",   # amber
    "Material":    "#ef4444",   # red
    "Other":       "#6b7280",   # gray
}


# ── DB helpers ────────────────────────────────────────────────────────────────

def _open_db() -> sqlite3.Connection:
    GRAPH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(GRAPH_DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_graph_db() -> None:
    conn = _open_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            entity_id   TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            name_lower  TEXT NOT NULL,
            type        TEXT NOT NULL DEFAULT 'Other',
            description TEXT NOT NULL DEFAULT '',
            paper_ids   TEXT NOT NULL DEFAULT '[]',   -- JSON list
            chunk_count INTEGER NOT NULL DEFAULT 0,
            created_at  REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_entities_name
            ON entities(name_lower);

        CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
            name,
            description,
            entity_id UNINDEXED
        );

        CREATE TABLE IF NOT EXISTS relations (
            relation_id TEXT PRIMARY KEY,
            source_id   TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
            target_id   TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
            relation    TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            paper_id    TEXT NOT NULL DEFAULT '',
            weight      REAL NOT NULL DEFAULT 1.0,
            created_at  REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id);
        CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id);

        CREATE TABLE IF NOT EXISTS entity_chunks (
            entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
            chunk_id  TEXT NOT NULL,
            paper_id  TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (entity_id, chunk_id)
        );
        CREATE INDEX IF NOT EXISTS idx_ec_chunk ON entity_chunks(chunk_id);

        CREATE TABLE IF NOT EXISTS communities (
            community_id TEXT PRIMARY KEY,
            entity_ids   TEXT NOT NULL DEFAULT '[]',   -- JSON list
            summary      TEXT NOT NULL DEFAULT '',
            created_at   REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS graph_meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    conn.commit()
    conn.close()


# ── Entity upsert ─────────────────────────────────────────────────────────────

def upsert_entity(
    conn: sqlite3.Connection,
    name: str,
    entity_type: str,
    description: str,
    paper_id: str,
) -> str:
    """Insert or update an entity; return entity_id."""
    name_lower = name.strip().lower()
    row = conn.execute(
        "SELECT entity_id, paper_ids FROM entities WHERE name_lower=?",
        (name_lower,),
    ).fetchone()

    if row:
        entity_id = row["entity_id"]
        paper_ids = set(json.loads(row["paper_ids"] or "[]"))
        paper_ids.add(paper_id)
        conn.execute(
            "UPDATE entities SET paper_ids=?, chunk_count=chunk_count+1 WHERE entity_id=?",
            (json.dumps(list(paper_ids)), entity_id),
        )
    else:
        entity_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO entities (entity_id, name, name_lower, type, description, paper_ids, chunk_count, created_at)"
            " VALUES (?,?,?,?,?,?,1,?)",
            (entity_id, name.strip(), name_lower, entity_type, description,
             json.dumps([paper_id]), time.time()),
        )
        conn.execute(
            "INSERT INTO entities_fts (name, description, entity_id) VALUES (?,?,?)",
            (name.strip(), description, entity_id),
        )
    return entity_id


def upsert_relation(
    conn: sqlite3.Connection,
    source_id: str,
    target_id: str,
    relation: str,
    description: str,
    paper_id: str,
) -> str:
    existing = conn.execute(
        "SELECT relation_id, weight FROM relations"
        " WHERE source_id=? AND target_id=? AND relation=?",
        (source_id, target_id, relation),
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE relations SET weight=weight+1 WHERE relation_id=?",
            (existing["relation_id"],),
        )
        return existing["relation_id"]
    rid = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO relations (relation_id, source_id, target_id, relation, description, paper_id, created_at)"
        " VALUES (?,?,?,?,?,?,?)",
        (rid, source_id, target_id, relation, description, paper_id, time.time()),
    )
    return rid


def link_entity_chunk(conn: sqlite3.Connection, entity_id: str, chunk_id: str, paper_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO entity_chunks (entity_id, chunk_id, paper_id) VALUES (?,?,?)",
        (entity_id, chunk_id, paper_id),
    )


# ── Graph queries ─────────────────────────────────────────────────────────────

def search_entities_fts(query: str, limit: int = 20) -> list[dict]:
    """FTS search on entity name + description."""
    conn = _open_db()
    safe = re.sub(r'["\(\)\*\:\^]', " ", query).strip()
    results = []
    if safe:
        try:
            rows = conn.execute(
                "SELECT e.entity_id, e.name, e.type, e.description, e.paper_ids, e.chunk_count"
                " FROM entities_fts f JOIN entities e ON e.entity_id = f.entity_id"
                " WHERE entities_fts MATCH ? LIMIT ?",
                (f'"{safe}"', limit),
            ).fetchall()
            results = [dict(r) for r in rows]
        except Exception:
            # fall back to LIKE search
            rows = conn.execute(
                "SELECT entity_id, name, type, description, paper_ids, chunk_count"
                " FROM entities WHERE name_lower LIKE ? LIMIT ?",
                (f"%{safe.lower()}%", limit),
            ).fetchall()
            results = [dict(r) for r in rows]
    conn.close()
    return results


def expand_entity_neighbors(entity_ids: list[str], hops: int = 1) -> list[dict]:
    """Return neighbour entity_ids reachable within `hops` from the seed set."""
    if not entity_ids:
        return []
    conn = _open_db()
    seen = set(entity_ids)
    frontier = set(entity_ids)

    for _ in range(hops):
        if not frontier:
            break
        placeholders = ",".join("?" * len(frontier))
        rows = conn.execute(
            f"SELECT DISTINCT source_id, target_id FROM relations"
            f" WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
            list(frontier) * 2,
        ).fetchall()
        new = set()
        for r in rows:
            for eid in (r["source_id"], r["target_id"]):
                if eid not in seen:
                    new.add(eid)
        seen |= new
        frontier = new

    if not seen:
        conn.close()
        return []

    placeholders = ",".join("?" * len(seen))
    entities = conn.execute(
        f"SELECT entity_id, name, type, description, paper_ids, chunk_count"
        f" FROM entities WHERE entity_id IN ({placeholders})",
        list(seen),
    ).fetchall()
    conn.close()
    return [dict(e) for e in entities]


def get_chunks_for_entities(entity_ids: list[str], limit: int = 30) -> list[str]:
    """Return chunk_ids linked to the given entity_ids, ordered by frequency."""
    if not entity_ids:
        return []
    conn = _open_db()
    placeholders = ",".join("?" * len(entity_ids))
    rows = conn.execute(
        f"SELECT chunk_id, COUNT(*) as cnt FROM entity_chunks"
        f" WHERE entity_id IN ({placeholders})"
        f" GROUP BY chunk_id ORDER BY cnt DESC LIMIT ?",
        entity_ids + [limit],
    ).fetchall()
    conn.close()
    return [r["chunk_id"] for r in rows]


def remove_paper(paper_id: str) -> dict:
    """Remove all graph traces of one paper.

    - delete its relations and entity_chunks rows
    - strip the paper_id from every entity's paper_ids list (and decrement
      chunk_count by the chunks contributed by this paper)
    - delete entities left with no papers (and their FTS rows; FK cascade clears
      any dangling relations/entity_chunks)

    Returns counts. Safe to call when the graph is empty / never built.
    """
    conn = _open_db()
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        # chunks this paper contributed per entity (for chunk_count fixups)
        per_entity = {
            r["entity_id"]: r["c"] for r in conn.execute(
                "SELECT entity_id, COUNT(*) AS c FROM entity_chunks"
                " WHERE paper_id=? GROUP BY entity_id", (paper_id,),
            ).fetchall()
        }
        conn.execute("DELETE FROM entity_chunks WHERE paper_id=?", (paper_id,))
        conn.execute("DELETE FROM relations WHERE paper_id=?", (paper_id,))

        removed_entities = updated_entities = 0
        rows = conn.execute(
            "SELECT entity_id, paper_ids, chunk_count FROM entities"
            " WHERE paper_ids LIKE ?", (f'%"{paper_id}"%',),
        ).fetchall()
        for r in rows:
            pids = [p for p in json.loads(r["paper_ids"] or "[]") if p != paper_id]
            if not pids:
                conn.execute("DELETE FROM entities WHERE entity_id=?", (r["entity_id"],))
                conn.execute("DELETE FROM entities_fts WHERE entity_id=?", (r["entity_id"],))
                removed_entities += 1
            else:
                new_count = max(0, (r["chunk_count"] or 0) - per_entity.get(r["entity_id"], 0))
                conn.execute(
                    "UPDATE entities SET paper_ids=?, chunk_count=? WHERE entity_id=?",
                    (json.dumps(pids), new_count, r["entity_id"]),
                )
                updated_entities += 1
        conn.commit()
        return {"entities_removed": removed_entities, "entities_updated": updated_entities}
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()


def get_graph_data(
    paper_ids: list[str] | None = None,
    entity_limit: int = 300,
    relation_limit: int = 500,
) -> dict:
    """Return {nodes, edges} for D3 force-directed visualization."""
    conn = _open_db()

    if paper_ids:
        # Filter entities that appear in these papers
        # paper_ids is stored as JSON array — use LIKE for simplicity
        conditions = " OR ".join(["paper_ids LIKE ?" for _ in paper_ids])
        params = [f'%"{p}"%' for p in paper_ids]
        entity_rows = conn.execute(
            f"SELECT entity_id, name, type, description, chunk_count"
            f" FROM entities WHERE ({conditions})"
            f" ORDER BY chunk_count DESC LIMIT ?",
            params + [entity_limit],
        ).fetchall()
    else:
        entity_rows = conn.execute(
            "SELECT entity_id, name, type, description, chunk_count"
            " FROM entities ORDER BY chunk_count DESC LIMIT ?",
            (entity_limit,),
        ).fetchall()

    entity_ids = [r["entity_id"] for r in entity_rows]
    nodes = [
        {
            "id": r["entity_id"],
            "name": r["name"],
            "type": r["type"],
            "description": r["description"],
            "weight": r["chunk_count"],
            "color": ENTITY_COLORS.get(r["type"], ENTITY_COLORS["Other"]),
        }
        for r in entity_rows
    ]

    edges = []
    if entity_ids:
        placeholders = ",".join("?" * len(entity_ids))
        rel_rows = conn.execute(
            f"SELECT source_id, target_id, relation, description, weight"
            f" FROM relations"
            f" WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders})"
            f" ORDER BY weight DESC LIMIT ?",
            entity_ids + entity_ids + [relation_limit],
        ).fetchall()
        edges = [
            {
                "source": r["source_id"],
                "target": r["target_id"],
                "relation": r["relation"],
                "description": r["description"],
                "weight": r["weight"],
            }
            for r in rel_rows
        ]

    conn.close()
    return {"nodes": nodes, "edges": edges}


def get_graph_stats() -> dict:
    conn = _open_db()
    entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    relations = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
    entity_chunks = conn.execute("SELECT COUNT(DISTINCT chunk_id) FROM entity_chunks").fetchone()[0]
    papers_indexed = conn.execute(
        "SELECT value FROM graph_meta WHERE key='papers_indexed'"
    ).fetchone()
    last_run = conn.execute(
        "SELECT value FROM graph_meta WHERE key='last_extraction_at'"
    ).fetchone()
    conn.close()
    return {
        "entities": entities,
        "relations": relations,
        "chunks_with_entities": entity_chunks,
        "papers_indexed": int(papers_indexed["value"]) if papers_indexed else 0,
        "last_extraction_at": float(last_run["value"]) if last_run else None,
    }


def set_graph_meta(key: str, value: str) -> None:
    conn = _open_db()
    conn.execute(
        "INSERT OR REPLACE INTO graph_meta (key, value) VALUES (?,?)",
        (key, value),
    )
    conn.commit()
    conn.close()
