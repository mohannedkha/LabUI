"""
RAG endpoints (FastAPI), mounted under prefix `/api/rag` from jfr/web/app.py.

Heavy modules (the bge-m3 embedder and BGE reranker) are loaded once in the
FastAPI lifespan handler (see jfr/web/app.py) and stashed on app.state; we mirror
them onto module-level globals here for the request handlers to use.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests
from fastapi import APIRouter, HTTPException, Query, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# ── Make Local_Rag/rag importable ─────────────────────────────────────────────
# Default to the sibling Local_Rag/rag tree, derived from this file's location so
# the app is portable; override with the LABUI_RAG_DIR env var if relocated.
_DEFAULT_RAG_DIR = (
    Path(__file__).resolve().parents[3] / "Local_Rag" / "rag"
)
_LOCAL_RAG_DIR = Path(
    os.environ.get("LABUI_RAG_DIR")
    or os.environ.get("CODEX_RAG_DIR")
    or str(_DEFAULT_RAG_DIR)
)
if str(_LOCAL_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_LOCAL_RAG_DIR))

from config import (  # noqa: E402  — comes from Local_Rag/rag/config.py
    OLLAMA_BASE_URL, GEN_MODEL, GEN_MODEL_ALIAS, GEN_MODEL_PREFER,
    DB_PATH, SUMMARIES_DIR, FINAL_TOP_K, NOTES_DIR, PARSED_DIR, DATA_DIR,
    PAPERS_PDF_DIR, INGEST_POLL_INTERVAL, RAG_ROOT,
    GEN_NUM_CTX, GEN_TEMPERATURE, GEN_TOP_P,
    STYLE_DIR, STYLE_MAX_CHARS,
    EMBED_MODEL_BASE, EMBED_DIM,
)
from retrieval.embed import load_embed_model, embed_query as _embed_query  # noqa: E402
from retrieval.rerank import load_reranker  # noqa: E402
from retrieval.search import search as _search  # noqa: E402
from generation.validate import validate_citations  # noqa: E402
from generation.citations import build_citation_map, build_references, strip_model_references  # noqa: E402
from generation.agents import AGENTS, get_agent, DEFAULT_AGENT  # noqa: E402
from retrieval.web_search import web_search as _web_search  # noqa: E402
import memory as _mem  # noqa: E402
from retrieval.graph import (  # noqa: E402
    init_graph_db as _init_graph_db,
    get_graph_stats as _graph_stats,
    get_graph_data as _get_graph_data,
    search_entities_fts as _search_entities,
    expand_entity_neighbors as _expand_neighbors,
    get_chunks_for_entities as _entity_chunks,
    remove_paper as _graph_remove_paper,
)


router = APIRouter()


# ── Module-level model handles (populated by lifespan handler) ────────────────
_tokenizer = None
_embed_model = None
_reranker = None


def set_models(tokenizer, embed_model, reranker) -> None:
    """Called from the FastAPI lifespan once models are loaded."""
    global _tokenizer, _embed_model, _reranker
    _tokenizer, _embed_model, _reranker = tokenizer, embed_model, reranker


def _get_models():
    return _tokenizer, _embed_model, _reranker


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def list_ollama_models() -> list[str]:
    """Names of models the user has actually pulled into Ollama (empty on error)."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        r.raise_for_status()
        return [m.get("name", "") for m in (r.json().get("models") or []) if m.get("name")]
    except Exception:
        return []


_resolved_gen_model: Optional[str] = None


def resolve_gen_model(force: bool = False) -> str:
    """Pick the generation model from whatever Ollama actually has installed —
    nothing is hardcoded. Order of preference:

      1. RAG_GEN_MODEL / CODEX_GEN_MODEL if set AND present in Ollama
         (or a tag whose base name matches, e.g. 'gemma3' → 'gemma3:12b').
      2. The first installed model matching RAG_GEN_PREFER substrings.
      3. The first installed model of any kind.

    Result is cached; pass force=True to re-resolve (e.g. after the user pulls a
    new model). Returns "" only when Ollama has no models at all.
    """
    global _resolved_gen_model
    if _resolved_gen_model and not force:
        return _resolved_gen_model

    names = list_ollama_models()
    chosen = ""
    if GEN_MODEL:
        if GEN_MODEL in names:
            chosen = GEN_MODEL
        else:
            base = GEN_MODEL.split(":")[0]
            chosen = next((n for n in names if n.split(":")[0] == base), "")
    if not chosen:
        for pref in GEN_MODEL_PREFER:
            match = next((n for n in names if pref.lower() in n.lower()), "")
            if match:
                chosen = match
                break
    if not chosen and names:
        chosen = names[0]

    _resolved_gen_model = chosen
    return chosen


def _list_pdf_files() -> list:
    """All PDFs in the papers dir, matched case-insensitively (.pdf / .PDF / …)
    and skipping macOS `._` sidecars. Python's glob('*.pdf') is case-sensitive,
    so a file saved as `Foo.PDF` would otherwise be invisible to the indexer."""
    if not PAPERS_PDF_DIR.exists():
        return []
    return sorted(
        p for p in PAPERS_PDF_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf" and not p.name.startswith("._")
    )


# ── Clips (saved figures + highlights from the in-app PDF reader) ─────────────
CLIPS_DIR = DATA_DIR / "clips"
CLIPS_DB = DATA_DIR / "clips.db"


def _open_clips_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(CLIPS_DB))
    conn.row_factory = sqlite3.Row
    return conn


def _init_clips_db() -> None:
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    conn = _open_clips_db()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS clips (
            clip_id       TEXT PRIMARY KEY,
            paper_id      TEXT NOT NULL,
            page          INTEGER DEFAULT 1,
            type          TEXT NOT NULL,          -- 'figure' | 'highlight'
            text          TEXT DEFAULT '',
            note          TEXT DEFAULT '',
            image_path    TEXT DEFAULT '',        -- relative to CLIPS_DIR
            rect          TEXT DEFAULT '',        -- JSON [x,y,w,h]
            manuscript_id TEXT DEFAULT '',
            created_at    REAL NOT NULL
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_clips_paper ON clips(paper_id)")
    conn.commit()
    conn.close()


def _new_clip_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_") + \
        base64.urlsafe_b64encode(os.urandom(4)).decode().rstrip("=")


def _delete_clips_for_paper(paper_id: str) -> int:
    """Remove a paper's clips + their image files (called from delete-paper)."""
    try:
        conn = _open_clips_db()
        rows = conn.execute("SELECT image_path FROM clips WHERE paper_id=?", (paper_id,)).fetchall()
        conn.execute("DELETE FROM clips WHERE paper_id=?", (paper_id,))
        conn.commit()
        conn.close()
    except Exception:
        return 0
    for r in rows:
        if r["image_path"]:
            try:
                (CLIPS_DIR / r["image_path"]).unlink(missing_ok=True)
            except Exception:
                pass
    return len(rows)


def _index_clip_chunk(clip_id: str, paper_id: str, text: str, page: int) -> bool:
    """Add a saved clip's text to the main search index as a citable chunk so the
    chat/search can retrieve your highlighted passages. No-op when the embedder
    isn't loaded or the paper isn't indexed."""
    text = (text or "").strip()
    if not text:
        return False
    tok, emb, _ = _get_models()
    if emb is None or not DB_PATH.exists():
        return False
    try:
        from retrieval.embed import embed_texts
        vec = embed_texts(tok, emb, [text])[0].astype("float32")
        conn = _open_db()
        try:
            if not conn.execute("SELECT 1 FROM papers WHERE paper_id=?", (paper_id,)).fetchone():
                return False
            chunk_id = f"clip_{clip_id}"
            conn.execute(
                "INSERT OR REPLACE INTO chunks"
                " (chunk_id, paper_id, section_name, page_start, page_end, position, text)"
                " VALUES (?,?,?,?,?,?,?)",
                (chunk_id, paper_id, "Saved highlight", page, page, 1000000, text),
            )
            conn.execute(
                "INSERT OR REPLACE INTO chunks_vec (chunk_id, embedding) VALUES (?,?)",
                (chunk_id, vec.tobytes()),
            )
            conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
            conn.commit()
            return True
        finally:
            conn.close()
    except Exception:
        return False


def _unindex_clip_chunk(clip_id: str) -> None:
    if not DB_PATH.exists():
        return
    try:
        conn = _open_db()
        cid = f"clip_{clip_id}"
        conn.execute("DELETE FROM chunks_vec WHERE chunk_id=?", (cid,))
        conn.execute("DELETE FROM chunks WHERE chunk_id=?", (cid,))
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        conn.commit()
        conn.close()
    except Exception:
        pass


def _existing_paper_ids() -> set[str]:
    """Paper IDs already in the index. Returns an empty set when the DB or its
    `papers` table doesn't exist yet — i.e. a fresh instance with nothing indexed.
    This is what lets the very first batch of uploaded PDFs index (build_index
    creates the DB), instead of the loop bailing out on a missing rag.db."""
    if not DB_PATH.exists():
        return set()
    try:
        conn = _open_db()
        try:
            return {row[0] for row in conn.execute("SELECT paper_id FROM papers").fetchall()}
        finally:
            conn.close()
    except sqlite3.Error:
        return set()


# ── Auto-index background thread (mirrors Flask server's behaviour) ───────────
_ingest_state: dict = {
    "running": False,
    "last_check": None,
    "new_count": 0,
    "message": "Starting…",
}


def _pdf_slug(stem: str) -> str:
    name = re.sub(r"[^\w\-]", "_", stem)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:120]


def _load_style_samples() -> Optional[str]:
    """Concatenate the user's writing-style samples (.md/.txt in STYLE_DIR),
    newest first, capped at STYLE_MAX_CHARS. Used to condition the writing agent."""
    if not STYLE_DIR.exists():
        return None
    files = [
        f for f in STYLE_DIR.iterdir()
        if f.is_file()
        and f.suffix.lower() in (".md", ".txt")
        and not f.name.startswith("._")
    ]
    if not files:
        return None
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    parts, total = [], 0
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if not text:
            continue
        block = f"--- sample: {f.stem} ---\n{text}"
        if total + len(block) > STYLE_MAX_CHARS:
            block = block[: max(0, STYLE_MAX_CHARS - total)]
        parts.append(block)
        total += len(block)
        if total >= STYLE_MAX_CHARS:
            break
    return "\n\n".join(parts) if parts else None


def _auto_index_loop() -> None:
    time.sleep(60)
    while True:
        try:
            _ingest_state["last_check"] = time.time()
            if not PAPERS_PDF_DIR.exists():
                _ingest_state["message"] = "Papers folder not found"
                time.sleep(INGEST_POLL_INTERVAL)
                continue

            # DB may not exist yet on a fresh instance — build_index creates it.
            existing_ids = _existing_paper_ids()

            pdf_files = _list_pdf_files()
            pdf_id_map = {_pdf_slug(p.stem): p for p in pdf_files}
            new_ids = set(pdf_id_map.keys()) - existing_ids
            _ingest_state["new_count"] = len(new_ids)

            if new_ids:
                _ingest_state["running"] = True
                _ingest_state["message"] = f"Indexing {len(new_ids)} new paper(s)…"
                _ingest_state["last_returncode"] = None

                def _tail(out: str, err: str, n: int = 180) -> str:
                    """Whichever stream is non-empty wins. Strips ANSI/blank lines."""
                    body = (err.strip() or out.strip() or "").splitlines()
                    body = [ln for ln in body if ln.strip()]
                    return (" │ ".join(body[-3:]) or "(no output)")[-n:]

                r1 = subprocess.run(
                    [sys.executable, "-m", "ingest.parse_pdfs"],
                    capture_output=True, text=True, cwd=str(RAG_ROOT),
                )
                if r1.returncode != 0:
                    _ingest_state["last_returncode"] = r1.returncode
                    _ingest_state["message"] = f"parse_pdfs failed (exit {r1.returncode}): {_tail(r1.stdout, r1.stderr)}"
                else:
                    r2 = subprocess.run(
                        [sys.executable, "-m", "ingest.build_index"],
                        capture_output=True, text=True, cwd=str(RAG_ROOT),
                    )
                    _ingest_state["last_returncode"] = r2.returncode
                    if r2.returncode != 0:
                        _ingest_state["message"] = f"build_index failed (exit {r2.returncode}): {_tail(r2.stdout, r2.stderr)}"
                    else:
                        _ingest_state["message"] = f"Indexed {len(new_ids)} paper(s) ✓"
                _ingest_state["running"] = False
            else:
                _ingest_state["message"] = f"Up to date ({len(pdf_files)} PDFs)"

        except Exception as exc:
            _ingest_state["message"] = f"Error: {exc}"
            _ingest_state["running"] = False

        time.sleep(INGEST_POLL_INTERVAL)


def start_auto_index() -> None:
    t = threading.Thread(target=_auto_index_loop, daemon=True, name="auto-index")
    t.start()


# ── Graph extraction background state ────────────────────────────────────────
_graph_state: dict = {
    "running": False, "progress": 0, "total": 0,
    "build_entities": 0, "build_relations": 0,
    "message": "Not started", "error": None,
}
_graph_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="graph-extract")


def _run_graph_extraction_bg() -> None:
    _graph_state.update({
        "running": True, "progress": 0, "total": 0,
        "message": "Starting…", "error": None,
    })

    def _cb(done, total, stats):
        _graph_state.update({
            "progress": done, "total": total,
            "build_entities": stats.get("entities", 0),
            "build_relations": stats.get("relations", 0),
            "message": f"{done}/{total} chunks processed",
        })

    try:
        from ingest.extract_graph import run_extraction
        stats = run_extraction(progress_cb=_cb)
        _graph_state.update({
            "running": False,
            "message": f"Done — {stats['entities']} entities, {stats['relations']} relations",
        })
    except SystemExit as e:
        # extract_graph calls sys.exit(1) when spacy/gliner are missing
        _graph_state.update({
            "running": False,
            "message": "Graph extractor dependencies missing (spacy / gliner / en_core_web_sm). Install them in the jfr venv.",
            "error": f"SystemExit({e.code})",
        })
    except BaseException as e:
        _graph_state.update({
            "running": False,
            "message": f"Error: {type(e).__name__}: {e}",
            "error": str(e),
        })


# ── Memory extraction thread (single worker) ─────────────────────────────────
_mem_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mem-extract")


def _async_extract_memory(query, answer, session_id, tok, emb,
                          base_url=OLLAMA_BASE_URL, gen_model=GEN_MODEL):
    try:
        _mem.extract_and_save_memories(
            query=query, answer=answer, session_id=session_id,
            tokenizer=tok, embed_model=emb,
            ollama_url=base_url, gen_model=gen_model,
        )
    except Exception as e:
        print(f"[memory] async extraction error: {e}")


# ── /agents ───────────────────────────────────────────────────────────────────
@router.get("/agents")
def api_agents():
    return {aid: {k: v for k, v in cfg.items() if k != "system"} for aid, cfg in AGENTS.items()}


# ── /models ───────────────────────────────────────────────────────────────────
@router.get("/models")
def api_models():
    """List Ollama models with metadata. Used by the in-app model picker."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        r.raise_for_status()
        tags = r.json().get("models", []) or []
    except Exception as e:
        return {"default": resolve_gen_model(), "models": [], "error": str(e)}

    default_model = resolve_gen_model(force=True)
    models = []
    for m in tags:
        details = m.get("details") or {}
        size = int(m.get("size") or 0)
        models.append({
            "name": m.get("name"),
            "modified_at": m.get("modified_at"),
            "size_bytes": size,
            "size_gb": round(size / 1_073_741_824, 1) if size else None,
            "family": details.get("family"),
            "parameter_size": details.get("parameter_size"),
            "quantization": details.get("quantization_level"),
            "is_default": m.get("name") == default_model,
        })
    # Sort: default first, then by parameter size descending (rough heuristic)
    def _sortkey(m):
        ps = (m.get("parameter_size") or "0").upper().rstrip("B")
        try:    n = float(ps)
        except: n = 0.0
        return (0 if m.get("is_default") else 1, -n, m.get("name") or "")
    models.sort(key=_sortkey)
    return {"default": default_model, "models": models}


# ── /status ───────────────────────────────────────────────────────────────────
@router.get("/status")
def api_status():
    resolved_model = resolve_gen_model(force=True)
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        tags = r.json().get("models", [])
        ollama_ok = r.status_code == 200
        model_names = [m.get("name", "") for m in tags]
        # "OK" = we have a usable generator (the resolved model is installed).
        gen_model_ok = bool(resolved_model) and resolved_model in model_names
    except Exception:
        ollama_ok = False
        gen_model_ok = False

    paper_count, chunk_count, index_ok = 0, 0, False
    if DB_PATH.exists():
        try:
            conn = _open_db()
            paper_count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            conn.close()
            index_ok = paper_count > 0
        except Exception:
            pass

    # bge-m3 (SentenceTransformer) has no separate tokenizer, so `tok` is None by
    # design — the embedder is `emb`. Checking tok here is what lit the red dot.
    _tok, emb, _ = _get_models()
    embed_ok = emb is not None

    return {
        "ollama": ollama_ok, "gen_model": gen_model_ok,
        "embeddings": embed_ok, "index": index_ok,
        "papers": paper_count, "chunks": chunk_count,
        "ingest": _ingest_state,
        "gen_model_name": resolved_model or "(no model installed)",
        "embed_model_name": EMBED_MODEL_BASE,
        "embed_dim": EMBED_DIM,
    }


# ── /search ───────────────────────────────────────────────────────────────────
class SearchBody(BaseModel):
    query: str
    top_k: int = 20
    selected_paper_ids: Optional[list[str]] = None


@router.post("/search")
def api_search(body: SearchBody):
    query = body.query.strip()
    if not query:
        raise HTTPException(400, "empty query")

    tok, emb, rnk = _get_models()
    if emb is None:
        raise HTTPException(503, "embedding model not loaded")

    try:
        qvec = _embed_query(tok, emb, query)
        results = _search(query, qvec, rnk, top_k=body.top_k,
                          selected_paper_ids=body.selected_paper_ids)
    except Exception as e:
        raise HTTPException(500, str(e))

    return {"results": results, "total": len(results)}


# ── /query (SSE streaming) ────────────────────────────────────────────────────
class QueryBody(BaseModel):
    query: str
    agent: Optional[str] = None
    session_id: Optional[str] = None
    base_url: Optional[str] = None
    gen_model: Optional[str] = None
    top_k: Optional[int] = None
    images: list[str] = []
    selected_paper_ids: Optional[list[str]] = None
    doc_context: Optional[str] = None


def _stream_tokens(query, chunks, agent_id=DEFAULT_AGENT, web_results=None,
                   images=None, doc_context=None, memories=None,
                   base_url=OLLAMA_BASE_URL, gen_model=GEN_MODEL,
                   style_samples=None):
    from generation.prompt import build_messages

    messages = build_messages(
        query, chunks, agent_id, web_results,
        doc_context=doc_context, memories=memories,
        style_samples=style_samples,
    )

    # Native /api/chat attaches images as a base64 list on the message (not as
    # OpenAI-style image_url content blocks). Content stays a plain string.
    if images:
        for msg in reversed(messages):
            if msg["role"] == "user":
                msg["images"] = list(images)
                break

    # Use the NATIVE /api/chat endpoint (not /v1/chat/completions) so we can pass
    # options.num_ctx. The OpenAI-compat endpoint silently ignores num_ctx, which
    # left long RAG prompts (8 chunks + web + memory) truncated at Ollama's
    # default context window and degraded answer/citation quality.
    payload = {
        "model": gen_model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": GEN_TEMPERATURE,
            "top_p": GEN_TOP_P,
            "num_ctx": GEN_NUM_CTX,
        },
    }
    url = f"{base_url}/api/chat"
    with requests.post(url, json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            token = obj.get("message", {}).get("content", "")
            if token:
                yield token
            if obj.get("done"):
                break


@router.post("/query")
def api_query(body: QueryBody):
    query = body.query.strip()
    agent_id = body.agent or DEFAULT_AGENT
    session_id = body.session_id
    base_url = body.base_url or OLLAMA_BASE_URL
    gen_model = body.gen_model or resolve_gen_model()
    agent = get_agent(agent_id)
    default_top_k = agent.get("top_k", FINAL_TOP_K)
    top_k = min(int(body.top_k or default_top_k), 12)
    images = body.images or []
    selected_paper_ids = body.selected_paper_ids
    doc_context = (body.doc_context or "").strip() or None
    cite_required = agent.get("cite_required", True)
    style_samples = _load_style_samples() if agent_id == "writing" else None

    if not query:
        raise HTTPException(400, "empty query")

    tok, emb, rnk = _get_models()
    if emb is None:
        raise HTTPException(503, "embedding model not loaded")

    # Parallel local retrieval + web search
    chunks, web_results = [], []
    try:
        qvec = _embed_query(tok, emb, query)
        with ThreadPoolExecutor(max_workers=2) as pool:
            local_fut = pool.submit(_search, query, qvec, rnk, top_k, selected_paper_ids)
            web_fut = pool.submit(_web_search, query, 6)
            chunks = local_fut.result()
            web_results = web_fut.result()
    except Exception as e:
        raise HTTPException(500, str(e))

    # Graph-augmented retrieval
    graph_entities: list[dict] = []
    try:
        seed_entities = _search_entities(query, limit=10)
        if seed_entities:
            expanded = _expand_neighbors([e["entity_id"] for e in seed_entities], hops=1)
            all_eids = list({e["entity_id"] for e in expanded})
            extra_chunk_ids = _entity_chunks(all_eids, limit=20)
            existing_chunk_ids = {c["chunk_id"] for c in chunks}
            new_cids = [cid for cid in extra_chunk_ids if cid not in existing_chunk_ids]
            if new_cids:
                conn = _open_db()
                conn.row_factory = sqlite3.Row
                placeholders = ",".join("?" * len(new_cids))
                extra_rows = conn.execute(
                    f"SELECT c.chunk_id, c.paper_id, c.text, c.section_name,"
                    f"       c.page_start, c.page_end, p.title, p.authors, p.year, p.doi"
                    f" FROM chunks c JOIN papers p USING(paper_id)"
                    f" WHERE c.chunk_id IN ({placeholders})",
                    new_cids,
                ).fetchall()
                conn.close()
                for row in extra_rows[:6]:
                    chunks.append(dict(row))
            graph_entities = seed_entities[:8]
    except Exception:
        pass

    # Memory retrieval
    memories = []
    try:
        memories = _mem.search_memories(qvec, query, top_k=5)
    except Exception:
        pass

    # Session bookkeeping
    is_new_session = not session_id
    if is_new_session:
        title = (query[:60] + "…") if len(query) > 60 else query
        session_id = _mem.create_session(title, agent_id)
    try:
        _mem.add_turn(session_id, "user", query, agent_id)
    except Exception:
        pass

    # One source card per paper, numbered to match the inline [n] citations and
    # the appended References list (see generation.citations.build_citation_map).
    cite_map = build_citation_map(chunks)
    _seen_pids: set[str] = set()
    papers_meta = []
    for c in chunks:
        pid = c["paper_id"]
        if pid in _seen_pids:
            continue
        _seen_pids.add(pid)
        papers_meta.append({
            "n": cite_map.get(pid),
            "paper_id": pid,
            "title": c.get("title", ""),
            "authors": c.get("authors", ""),
            "year": c.get("year"),
            "doi": c.get("doi"),
            "section_name": c.get("section_name", ""),
            "page_start": c.get("page_start"),
            "page_end": c.get("page_end"),
        })
    papers_meta.sort(key=lambda p: p["n"] or 0)

    def gen():
        yield (
            f"event: session\n"
            f"data: {json.dumps({'session_id': session_id, 'is_new': is_new_session})}\n\n"
        )
        yield f"event: papers\ndata: {json.dumps({'papers': papers_meta, 'agent': agent_id})}\n\n"
        yield f"event: web_results\ndata: {json.dumps(web_results)}\n\n"
        if graph_entities:
            yield f"event: graph_entities\ndata: {json.dumps(graph_entities)}\n\n"
        if memories:
            yield (
                f"event: memories\n"
                f"data: {json.dumps([{'content': m['content'], 'memory_type': m['memory_type']} for m in memories])}\n\n"
            )

        try:
            collected = []
            for token in _stream_tokens(
                query, chunks, agent_id, web_results, images, doc_context, memories,
                base_url, gen_model, style_samples,
            ):
                collected.append(token)
                yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"

            raw_text = "".join(collected)

            # Drop any reference list the model wrote on its own, then append our
            # numbered (ACS-style) References built from the cited sources. The
            # cleaned text is sent in `done.answer` so the UI renders the final
            # version (it streamed the raw tokens live).
            answer = strip_model_references(raw_text) if cite_required else raw_text
            references = build_references(chunks, answer) if cite_required else ""
            validated = validate_citations(answer, chunks, cite_required=cite_required)
            warning = validated[len(answer):]
            final_text = answer + references + warning

            yield (
                f"event: done\n"
                f"data: {json.dumps({'citations_ok': not warning, 'answer': final_text})}\n\n"
            )

            try:
                _mem.add_turn(
                    session_id, "assistant", answer + references, agent_id,
                    sources=papers_meta, web=web_results,
                )
            except Exception:
                pass
            _mem_executor.submit(_async_extract_memory, query, answer, session_id, tok, emb, base_url, gen_model)

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── /paper ────────────────────────────────────────────────────────────────────
@router.get("/paper")
def api_paper(id: str = Query("")):
    paper_id = id.strip()
    if not paper_id:
        raise HTTPException(400, "missing id parameter")
    if not DB_PATH.exists():
        raise HTTPException(404, "database not found")

    try:
        conn = _open_db()
        row = conn.execute(
            "SELECT title, authors, year, source_pdf FROM papers WHERE paper_id = ?",
            (paper_id,),
        ).fetchone()
        sections = conn.execute(
            "SELECT section_name, page_start, page_end FROM chunks "
            "WHERE paper_id = ? GROUP BY section_name ORDER BY MIN(position)",
            (paper_id,),
        ).fetchall()
        conn.close()
    except Exception as e:
        raise HTTPException(500, str(e))

    if not row:
        raise HTTPException(404, "paper not found")

    summary_text = ""
    try:
        for md in SUMMARIES_DIR.glob("*.md"):
            if paper_id in md.stem or paper_id.replace("_", " ") in md.stem:
                summary_text = md.read_text(encoding="utf-8", errors="replace")
                break
    except Exception:
        pass

    pdf_available = bool(row[3] and Path(row[3]).exists())
    return {
        "paper_id": paper_id, "title": row[0], "authors": row[1], "year": row[2],
        "source_pdf": row[3], "pdf_available": pdf_available,
        "sections": [{"name": s[0], "page_start": s[1], "page_end": s[2]} for s in sections],
        "summary": summary_text,
    }


# ── /pdf ──────────────────────────────────────────────────────────────────────
@router.get("/pdf")
def api_pdf(id: str = Query("")):
    paper_id = id.strip()
    if not paper_id:
        raise HTTPException(400, "missing id")
    if not DB_PATH.exists():
        raise HTTPException(404, "database not found")

    try:
        conn = _open_db()
        row = conn.execute("SELECT source_pdf FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        conn.close()
    except Exception as e:
        raise HTTPException(500, str(e))

    if row and row[0]:
        p = Path(row[0])
        if p.exists():
            return FileResponse(str(p), media_type="application/pdf", filename=p.name)
    raise HTTPException(404, "PDF not found")


# ── /paper (DELETE) ────────────────────────────────────────────────────────────
@router.delete("/paper/{paper_id}")
def api_paper_delete(paper_id: str):
    """Remove a paper and ALL its index traces: chunks (+ FTS + vectors), the
    papers row, its parsed JSON, the source PDF (so the auto-indexer won't re-add
    it), and its knowledge-graph entries."""
    paper_id = (paper_id or "").strip()
    if not re.match(r"^[\w\-]+$", paper_id):
        raise HTTPException(400, "invalid id")
    if not DB_PATH.exists():
        raise HTTPException(404, "database not found")

    conn = _open_db()
    try:
        row = conn.execute("SELECT source_pdf FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
        if not row:
            raise HTTPException(404, "paper not found")
        source_pdf = row[0]
        chunk_ids = [r[0] for r in conn.execute(
            "SELECT chunk_id FROM chunks WHERE paper_id=?", (paper_id,)).fetchall()]
        if chunk_ids:
            conn.executemany("DELETE FROM chunks_vec WHERE chunk_id=?", [(c,) for c in chunk_ids])
        conn.execute("DELETE FROM chunks WHERE paper_id=?", (paper_id,))
        conn.execute("DELETE FROM papers WHERE paper_id=?", (paper_id,))
        # External-content FTS5: rebuild to drop the deleted chunks' terms.
        try:
            conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        except Exception:
            pass
        conn.commit()
    except HTTPException:
        conn.close()
        raise
    except Exception as e:
        conn.close()
        raise HTTPException(500, str(e))
    conn.close()

    # parsed JSON
    try:
        (PARSED_DIR / f"{paper_id}.json").unlink(missing_ok=True)
    except Exception:
        pass

    # source PDF — remove so the auto-index daemon doesn't re-ingest it
    pdf_removed = False
    try:
        if source_pdf and Path(source_pdf).exists():
            Path(source_pdf).unlink()
            pdf_removed = True
    except Exception:
        pass

    # knowledge graph
    try:
        graph_result = _graph_remove_paper(paper_id)
    except Exception as e:
        graph_result = {"error": str(e)}

    # saved clips (figures/highlights) for this paper
    clips_removed = _delete_clips_for_paper(paper_id)

    return {
        "ok": True, "paper_id": paper_id,
        "chunks_removed": len(chunk_ids),
        "pdf_removed": pdf_removed,
        "graph": graph_result,
        "clips_removed": clips_removed,
    }


# ── /upload ───────────────────────────────────────────────────────────────────
@router.post("/papers/upload")
async def api_papers_upload(files: list[UploadFile] = File(...)):
    """
    Upload one or more PDFs into PAPERS_PDF_DIR. The auto-index daemon picks
    them up within ~INGEST_POLL_INTERVAL seconds and indexes them. Returns
    per-file status.
    """
    PAPERS_PDF_DIR.mkdir(parents=True, exist_ok=True)
    MAX_BYTES = 200 * 1024 * 1024  # 200 MB per file
    results: list[dict] = []

    for upload in files:
        result: dict = {"filename": upload.filename, "ok": False}
        try:
            data = await upload.read()
            if not data:
                result["error"] = "empty file"
                results.append(result); continue
            if len(data) > MAX_BYTES:
                result["error"] = f"file > {MAX_BYTES // 1024 // 1024} MB cap"
                results.append(result); continue
            if not data.startswith(b"%PDF-"):
                result["error"] = "not a valid PDF (missing %PDF- magic bytes)"
                results.append(result); continue

            # Sanitize filename: drop directory parts, ensure .pdf extension
            raw_name = upload.filename or "upload.pdf"
            base = os.path.basename(raw_name)  # strip any path
            base = re.sub(r"[^\w\.\-\s -￿]", "_", base).strip()
            # Normalize the extension to lowercase `.pdf` (a saved `Foo.PDF` is
            # otherwise skipped by the indexer's case-sensitive matching).
            if base.lower().endswith(".pdf"):
                base = base[:-4] + ".pdf"
            else:
                base = base + ".pdf"
            base = base[:200]  # cap length
            dest = PAPERS_PDF_DIR / base

            # Collision handling
            if dest.exists():
                stem = dest.stem
                ext = dest.suffix
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                dest = dest.with_name(f"{stem}_{ts}{ext}")

            dest.write_bytes(data)
            result.update({
                "ok": True,
                "saved_as": dest.name,
                "bytes": len(data),
                "path": str(dest),
            })
        except Exception as e:
            result["error"] = str(e)
        results.append(result)

    # Nudge the auto-index loop so the user doesn't wait the full poll interval.
    # _ingest_state is just informational; the daemon will detect new files
    # on its own. We update the message immediately so the UI status reflects it.
    if any(r["ok"] for r in results):
        _ingest_state["new_count"] = sum(1 for r in results if r["ok"])
        _ingest_state["message"] = (
            f"{_ingest_state['new_count']} new paper(s) just uploaded — "
            f"auto-indexer will run shortly"
        )

    ok = sum(1 for r in results if r["ok"])
    return {
        "uploaded": ok,
        "failed": len(results) - ok,
        "results": results,
        "papers_dir": str(PAPERS_PDF_DIR),
        "auto_index_interval_seconds": INGEST_POLL_INTERVAL,
    }


@router.post("/upload")
async def api_upload(file: UploadFile = File(...)):
    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    IMAGE_EXTS = {"jpg", "jpeg", "png", "gif", "webp", "bmp"}
    data = await file.read()

    if ext in IMAGE_EXTS:
        b64 = base64.b64encode(data).decode("ascii")
        mime = f"image/{'jpeg' if ext == 'jpg' else ext}"
        return {"type": "image", "filename": filename, "data": b64, "mime": mime}

    if ext == "pdf":
        try:
            import fitz
            doc = fitz.open(stream=data, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)[:12000]
            doc.close()
            return {"type": "document", "filename": filename, "text": text}
        except Exception as e:
            raise HTTPException(500, f"PDF extraction failed: {e}")

    try:
        text = data.decode("utf-8", errors="replace")[:12000]
        return {"type": "document", "filename": filename, "text": text}
    except Exception:
        raise HTTPException(400, "unsupported file type")


# ── /notes ────────────────────────────────────────────────────────────────────
@router.get("/notes")
def api_notes_list():
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    notes = []
    for f in sorted(NOTES_DIR.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True):
        if f.name.startswith("._"):
            continue  # skip macOS AppleDouble sidecars
        content = f.read_text(encoding="utf-8", errors="replace")
        title = f.stem
        for line in content.split("\n"):
            if line.startswith("title:"):
                title = line[6:].strip().strip("\"'"); break
            if line.startswith("# "):
                title = line[2:].strip(); break
        notes.append({
            "id": f.stem, "title": title,
            "modified": f.stat().st_mtime,
            "preview": content[:200].replace("\n", " "),
        })
    return notes


class NoteCreate(BaseModel):
    title: str = "Untitled Note"
    content: str = ""
    agent: str = ""


@router.post("/notes")
def api_notes_create(body: NoteCreate):
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    title = (body.title or "Untitled Note").strip()
    content = (body.content or "").strip()
    agent = (body.agent or "").strip()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^\w\-]", "_", title[:40]).strip("_") or "note"
    note_id = f"{ts}_{safe}"
    fm = f"---\ntitle: {title}\nagent: {agent}\ndate: {datetime.now().isoformat()[:19]}\n---\n\n"
    (NOTES_DIR / f"{note_id}.md").write_text(fm + f"# {title}\n\n" + content, encoding="utf-8")
    return {"id": note_id, "title": title}


@router.get("/notes/{note_id}")
def api_note_get(note_id: str):
    if not re.match(r"^[\w\-]+$", note_id):
        raise HTTPException(400, "invalid id")
    p = NOTES_DIR / f"{note_id}.md"
    if not p.exists():
        raise HTTPException(404, "not found")
    return {"id": note_id, "content": p.read_text(encoding="utf-8", errors="replace")}


@router.delete("/notes/{note_id}")
def api_note_delete(note_id: str):
    if not re.match(r"^[\w\-]+$", note_id):
        raise HTTPException(400, "invalid id")
    p = NOTES_DIR / f"{note_id}.md"
    if not p.exists():
        raise HTTPException(404, "not found")
    p.unlink()
    return {"ok": True}


# ── /clips — saved figures + highlights from the in-app PDF reader ─────────────
class ClipCreate(BaseModel):
    paper_id: str
    page: int = 1
    type: str = "highlight"          # 'figure' | 'highlight'
    text: str = ""
    note: str = ""
    rect: Optional[list] = None
    image_b64: Optional[str] = None  # PNG data-URL or raw base64 (figure clips)
    manuscript_id: str = ""          # optional: attach to a manuscript


@router.post("/clips")
def api_clip_create(body: ClipCreate):
    pid = (body.paper_id or "").strip()
    if not pid:
        raise HTTPException(400, "paper_id required")
    ctype = body.type if body.type in ("figure", "highlight") else "highlight"
    clip_id = _new_clip_id()
    image_path = ""

    if ctype == "figure":
        if not body.image_b64:
            raise HTTPException(400, "figure clip requires image_b64")
        try:
            raw = base64.b64decode(body.image_b64.split(",", 1)[-1])
        except Exception:
            raise HTTPException(400, "invalid image data")
        (CLIPS_DIR / pid).mkdir(parents=True, exist_ok=True)
        (CLIPS_DIR / pid / f"{clip_id}.png").write_bytes(raw)
        image_path = f"{pid}/{clip_id}.png"
    elif not (body.text or "").strip():
        raise HTTPException(400, "highlight clip requires text")

    conn = _open_clips_db()
    conn.execute(
        "INSERT INTO clips (clip_id, paper_id, page, type, text, note, image_path,"
        " rect, manuscript_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (clip_id, pid, int(body.page or 1), ctype, (body.text or "").strip(),
         (body.note or "").strip(), image_path, json.dumps(body.rect or []),
         (body.manuscript_id or "").strip(), time.time()),
    )
    conn.commit()
    conn.close()

    # Make it searchable: highlight text (+ note), or a figure's caption.
    index_text = (body.text or "").strip()
    if (body.note or "").strip():
        index_text = (index_text + "\n\n" + body.note.strip()).strip()
    indexed = _index_clip_chunk(clip_id, pid, index_text, int(body.page or 1))

    return {"clip_id": clip_id, "type": ctype, "image_path": image_path, "indexed": indexed}


@router.get("/clips")
def api_clips_list(paper_id: str = Query(""), manuscript_id: str = Query("")):
    conn = _open_clips_db()
    if paper_id.strip():
        rows = conn.execute(
            "SELECT * FROM clips WHERE paper_id=? ORDER BY created_at DESC",
            (paper_id.strip(),),
        ).fetchall()
    elif manuscript_id.strip():
        rows = conn.execute(
            "SELECT * FROM clips WHERE manuscript_id=? ORDER BY created_at DESC",
            (manuscript_id.strip(),),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM clips ORDER BY created_at DESC LIMIT 500").fetchall()
    conn.close()
    return [dict(r) for r in rows]


class ClipPatch(BaseModel):
    manuscript_id: str = ""


@router.patch("/clips/{clip_id}")
def api_clip_patch(clip_id: str, body: ClipPatch):
    if not re.match(r"^[\w\-]+$", clip_id):
        raise HTTPException(400, "invalid id")
    conn = _open_clips_db()
    if not conn.execute("SELECT 1 FROM clips WHERE clip_id=?", (clip_id,)).fetchone():
        conn.close()
        raise HTTPException(404, "not found")
    conn.execute("UPDATE clips SET manuscript_id=? WHERE clip_id=?",
                 ((body.manuscript_id or "").strip(), clip_id))
    conn.commit()
    conn.close()
    return {"ok": True, "manuscript_id": (body.manuscript_id or "").strip()}


@router.get("/clips/{clip_id}/image")
def api_clip_image(clip_id: str):
    if not re.match(r"^[\w\-]+$", clip_id):
        raise HTTPException(400, "invalid id")
    conn = _open_clips_db()
    row = conn.execute("SELECT image_path FROM clips WHERE clip_id=?", (clip_id,)).fetchone()
    conn.close()
    if not row or not row["image_path"]:
        raise HTTPException(404, "no image for this clip")
    p = CLIPS_DIR / row["image_path"]
    if not p.exists():
        raise HTTPException(404, "image file missing")
    return FileResponse(str(p), media_type="image/png")


@router.delete("/clips/{clip_id}")
def api_clip_delete(clip_id: str):
    if not re.match(r"^[\w\-]+$", clip_id):
        raise HTTPException(400, "invalid id")
    conn = _open_clips_db()
    row = conn.execute("SELECT image_path FROM clips WHERE clip_id=?", (clip_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "not found")
    conn.execute("DELETE FROM clips WHERE clip_id=?", (clip_id,))
    conn.commit()
    conn.close()
    if row["image_path"]:
        try:
            (CLIPS_DIR / row["image_path"]).unlink(missing_ok=True)
        except Exception:
            pass
    _unindex_clip_chunk(clip_id)
    return {"ok": True}


# ── /style — writing-style samples for the Style Writer agent ─────────────────
@router.get("/style")
def api_style_list():
    STYLE_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for f in sorted(STYLE_DIR.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
        if not f.is_file() or f.suffix.lower() not in (".md", ".txt") or f.name.startswith("._"):
            continue
        content = f.read_text(encoding="utf-8", errors="replace")
        out.append({
            "id": f.stem,
            "title": f.stem,
            "chars": len(content),
            "modified": f.stat().st_mtime,
            "preview": content[:200].replace("\n", " "),
        })
    return out


class StyleCreate(BaseModel):
    title: str = "sample"
    content: str = ""


@router.post("/style")
def api_style_create(body: StyleCreate):
    content = (body.content or "").strip()
    if not content:
        raise HTTPException(400, "empty content")
    STYLE_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-]", "_", (body.title or "sample")[:60]).strip("_") or "sample"
    dest = STYLE_DIR / f"{safe}.md"
    if dest.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = STYLE_DIR / f"{safe}_{ts}.md"
    dest.write_text(content, encoding="utf-8")
    return {"id": dest.stem, "chars": len(content)}


@router.get("/style/{sample_id}")
def api_style_get(sample_id: str):
    if not re.match(r"^[\w\-]+$", sample_id):
        raise HTTPException(400, "invalid id")
    for ext in (".md", ".txt"):
        p = STYLE_DIR / f"{sample_id}{ext}"
        if p.exists():
            return {"id": sample_id, "content": p.read_text(encoding="utf-8", errors="replace")}
    raise HTTPException(404, "not found")


@router.delete("/style/{sample_id}")
def api_style_delete(sample_id: str):
    if not re.match(r"^[\w\-]+$", sample_id):
        raise HTTPException(400, "invalid id")
    for ext in (".md", ".txt"):
        p = STYLE_DIR / f"{sample_id}{ext}"
        if p.exists():
            p.unlink()
            return {"ok": True}
    raise HTTPException(404, "not found")


# ── /ingest/status ────────────────────────────────────────────────────────────
@router.get("/ingest/status")
def api_ingest_status():
    return _ingest_state


# ── /ingest/run — trigger an immediate index run ──────────────────────────────
_ingest_trigger_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest-trigger")


def _trigger_ingest_once() -> None:
    """Run parse_pdfs + build_index once, write state. Background-thread safe."""
    try:
        if not PAPERS_PDF_DIR.exists():
            _ingest_state.update({"running": False, "message": "Papers folder not found"})
            return

        # Detect new vs existing (DB may not exist yet on a fresh instance —
        # build_index creates it from scratch).
        existing_ids = _existing_paper_ids()
        pdf_files = _list_pdf_files()
        new_ids = {_pdf_slug(p.stem) for p in pdf_files} - existing_ids

        if not new_ids:
            _ingest_state.update({"running": False, "message": f"Up to date ({len(pdf_files)} PDFs)"})
            return

        _ingest_state.update({"running": True, "new_count": len(new_ids),
                              "message": f"Manually indexing {len(new_ids)} new paper(s)…"})

        def _tail(out: str, err: str, n: int = 180) -> str:
            body = (err.strip() or out.strip() or "").splitlines()
            body = [ln for ln in body if ln.strip()]
            return (" │ ".join(body[-3:]) or "(no output)")[-n:]

        r1 = subprocess.run([sys.executable, "-m", "ingest.parse_pdfs"],
                            capture_output=True, text=True, cwd=str(RAG_ROOT))
        if r1.returncode != 0:
            _ingest_state.update({"running": False, "last_returncode": r1.returncode,
                                  "message": f"parse_pdfs failed (exit {r1.returncode}): {_tail(r1.stdout, r1.stderr)}"})
            return
        r2 = subprocess.run([sys.executable, "-m", "ingest.build_index"],
                            capture_output=True, text=True, cwd=str(RAG_ROOT))
        if r2.returncode != 0:
            _ingest_state.update({"running": False, "last_returncode": r2.returncode,
                                  "message": f"build_index failed (exit {r2.returncode}): {_tail(r2.stdout, r2.stderr)}"})
            return
        _ingest_state.update({"running": False, "last_returncode": 0,
                              "message": f"Indexed {len(new_ids)} paper(s) ✓"})
    except Exception as e:
        _ingest_state.update({"running": False, "message": f"Trigger error: {e}"})


@router.post("/ingest/run")
def api_ingest_run():
    """Kick the indexer immediately instead of waiting for the next poll."""
    if _ingest_state.get("running"):
        raise HTTPException(409, "Indexer already running; wait for it to finish")
    _ingest_trigger_executor.submit(_trigger_ingest_once)
    return {"status": "started"}


# ── /sessions ─────────────────────────────────────────────────────────────────
@router.get("/sessions")
def api_sessions_list():
    return _mem.get_sessions(limit=80)


@router.get("/sessions/{session_id}")
def api_session_get(session_id: str):
    s = _mem.get_session(session_id)
    if not s:
        raise HTTPException(404, "not found")
    return s


@router.delete("/sessions/{session_id}")
def api_session_delete(session_id: str):
    _mem.delete_session(session_id)
    return {"ok": True}


class SessionRename(BaseModel):
    title: str


@router.put("/sessions/{session_id}/title")
def api_session_rename(session_id: str, body: SessionRename):
    title = (body.title or "").strip()
    if not title:
        raise HTTPException(400, "missing title")
    _mem.rename_session(session_id, title)
    return {"ok": True}


# ── /memory ───────────────────────────────────────────────────────────────────
@router.get("/memory")
def api_memory_list():
    return _mem.get_all_memories(limit=200)


@router.delete("/memory/{memory_id}")
def api_memory_delete(memory_id: str):
    _mem.delete_memory(memory_id)
    return {"ok": True}


class MemSearch(BaseModel):
    query: str


@router.post("/memory/search")
def api_memory_search(body: MemSearch):
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(400, "missing query")
    tok, emb, _ = _get_models()
    if emb is None:
        raise HTTPException(503, "embedding model not loaded")
    qvec = _embed_query(tok, emb, query)
    return _mem.search_memories(qvec, query, top_k=10)


# ── /graph ────────────────────────────────────────────────────────────────────
@router.get("/graph/stats")
def api_graph_stats():
    try:
        stats = _graph_stats()
    except Exception as e:
        stats = {"error": str(e)}
    # DB stats must win for entities/relations/papers — _graph_state's running-build
    # placeholders would otherwise zero them out at idle.
    return {**_graph_state, **stats}


class GraphBuild(BaseModel):
    reset: bool = False
    force: bool = False  # clear a stuck running=True flag from a prior crash


@router.post("/graph/build")
def api_graph_build(body: GraphBuild):
    if _graph_state["running"] and not body.force:
        raise HTTPException(409, "Extraction already running; pass force=true to reset")
    if body.force:
        _graph_state.update({"running": False, "message": "Reset", "error": None})
    if body.reset:
        from ingest.extract_graph import reset_graph_db
        reset_graph_db()
    _graph_executor.submit(_run_graph_extraction_bg)
    return {"status": "started"}


@router.get("/graph/data")
def api_graph_data(request: Request, limit: int = 300):
    paper_ids = request.query_params.getlist("paper_id") or None
    try:
        return _get_graph_data(paper_ids=paper_ids, entity_limit=limit)
    except Exception as e:
        return JSONResponse({"error": str(e), "nodes": [], "edges": []}, status_code=500)


@router.get("/graph/search")
def api_graph_search(q: str = ""):
    q = q.strip()
    if not q:
        return []
    return _search_entities(q, limit=20)


@router.get("/graph/entity/{entity_id}")
def api_graph_entity(entity_id: str, neighbor_limit: int = 20, chunk_limit: int = 8):
    """Entity detail: own metadata + 1-hop neighbors + top chunks (enriched)."""
    from retrieval.graph import _open_db as _open_graph_db
    g = _open_graph_db()
    row = g.execute(
        "SELECT entity_id, name, type, description, paper_ids, chunk_count"
        " FROM entities WHERE entity_id = ?", (entity_id,),
    ).fetchone()
    if not row:
        g.close()
        raise HTTPException(404, "entity not found")
    entity = dict(row)

    # 1-hop neighbors ordered by relation weight
    neighbour_rows = g.execute(
        """
        SELECT
          CASE WHEN r.source_id = ? THEN r.target_id ELSE r.source_id END AS nid,
          MAX(r.weight) AS weight,
          GROUP_CONCAT(DISTINCT r.relation) AS relations
        FROM relations r
        WHERE (r.source_id = ? OR r.target_id = ?)
          AND r.source_id != r.target_id
        GROUP BY nid
        ORDER BY weight DESC
        LIMIT ?
        """,
        (entity_id, entity_id, entity_id, neighbor_limit),
    ).fetchall()
    neighbour_ids = [r["nid"] for r in neighbour_rows]

    neighbours = []
    if neighbour_ids:
        placeholders = ",".join("?" * len(neighbour_ids))
        meta = g.execute(
            f"SELECT entity_id, name, type, description, chunk_count"
            f" FROM entities WHERE entity_id IN ({placeholders})",
            neighbour_ids,
        ).fetchall()
        by_id = {m["entity_id"]: dict(m) for m in meta}
        for r in neighbour_rows:
            m = by_id.get(r["nid"])
            if m:
                m["weight"] = r["weight"]
                m["relations"] = (r["relations"] or "").split(",")[:3]
                neighbours.append(m)

    chunk_ids = _entity_chunks([entity_id], limit=chunk_limit)
    g.close()

    # Enrich chunks from rag.db (chunks table + papers table)
    chunks = []
    if chunk_ids:
        try:
            conn = _open_db()
            conn.row_factory = sqlite3.Row
            placeholders = ",".join("?" * len(chunk_ids))
            rows = conn.execute(
                f"SELECT c.chunk_id, c.paper_id, c.section_name, c.page_start, c.page_end,"
                f"       c.text, p.title, p.authors, p.year"
                f"  FROM chunks c JOIN papers p USING(paper_id)"
                f" WHERE c.chunk_id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
            conn.close()
            order = {cid: i for i, cid in enumerate(chunk_ids)}
            for r in sorted((dict(r) for r in rows), key=lambda d: order.get(d["chunk_id"], 999)):
                snippet = (r.get("text") or "")[:320]
                chunks.append({
                    "chunk_id": r["chunk_id"],
                    "paper_id": r["paper_id"],
                    "title": r.get("title", ""),
                    "authors": r.get("authors", ""),
                    "year": r.get("year"),
                    "section_name": r.get("section_name", ""),
                    "page_start": r.get("page_start"),
                    "page_end": r.get("page_end"),
                    "snippet": snippet,
                })
        except Exception as e:
            chunks = [{"error": str(e)}]

    return {
        "entity": entity,
        "neighbours": neighbours,
        "chunks": chunks,
    }


def _sweep_macos_sidecars() -> int:
    """
    Delete `._*` AppleDouble files and .DS_Store left over from Mac→Linux copies.
    They break json.load / sentence-transformers / glob iteration in the RAG
    pipeline. Called once at startup so future ingests don't have to.
    """
    import os
    targets = [
        PAPERS_PDF_DIR,
        NOTES_DIR,
        STYLE_DIR,
        RAG_ROOT / "data",
    ]
    n = 0
    for root in targets:
        if not root.exists():
            continue
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.startswith("._") or f == ".DS_Store":
                    try:
                        (Path(dirpath) / f).unlink()
                        n += 1
                    except Exception:
                        pass
    if n:
        print(f"[rag] swept {n} macOS sidecar files from RAG data dirs")
    return n


# ── DB init helper used by lifespan ───────────────────────────────────────────
def init_databases() -> None:
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    STYLE_DIR.mkdir(parents=True, exist_ok=True)
    _sweep_macos_sidecars()
    _mem.init_db()
    _init_graph_db()
    _init_clips_db()
