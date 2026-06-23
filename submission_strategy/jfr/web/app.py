"""FastAPI web application — served on localhost:8765.

Start with:  jfr web serve
Or directly: uvicorn jfr.web.app:app --host 127.0.0.1 --port 8765 --reload
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from jfr.config import get_settings, load_policy
from jfr.db import get_conn
from jfr.db.schema import VALID_TRANSITIONS, TERMINAL_STATES


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load RAG models (SPECTER2 + reranker) once at startup; non-fatal on failure."""
    # Apply JFR schema (creates new tables like `experiment` idempotently)
    try:
        from jfr.config import get_settings as _gs
        from jfr.db.schema import init_db as _init_jfr_db
        _s = _gs()
        _conn0 = _init_jfr_db(_s.db_path)
        print(f"[jfr] schema applied at {_s.db_path}")
        # First-boot seed: load the starter journal set if the table is empty so a
        # fresh install has journals out of the box (users add more via /journals).
        try:
            n = _conn0.execute("SELECT COUNT(*) FROM journal").fetchone()[0]
            if n == 0 and _s.journals_yaml.exists():
                from jfr.tracker import load_journals_from_yaml
                loaded = load_journals_from_yaml(_conn0, _s.journals_yaml)
                print(f"[jfr] seeded {loaded} journals from {_s.journals_yaml.name}")
        except Exception as e:
            print(f"[jfr] journal seed skipped: {e}")
        _conn0.close()
    except Exception as e:
        print(f"[jfr] schema apply FAILED: {e}")

    try:
        from jfr.web import rag_routes
        print("[rag] initializing databases…")
        rag_routes.init_databases()
        print("[rag] loading BGE-M3 embedder…")
        from retrieval.embed import load_embed_model, _DEVICE as embed_dev
        tok, mdl = load_embed_model()
        print(f"[rag] embed model ready on {embed_dev}")
        print("[rag] loading BGE reranker…")
        from retrieval.rerank import load_reranker, _DEVICE as rk_dev
        rk = load_reranker()
        print(f"[rag] reranker ready on {rk_dev}")
        rag_routes.set_models(tok, mdl, rk)
        rag_routes.start_auto_index()
        app.state.rag_ready = True
    except Exception as e:
        print(f"[rag] startup FAILED (jfr still usable, RAG endpoints will 503): {e}")
        app.state.rag_ready = False

    # Journal corpus: weekly scheduled refresh + auto-populate when empty.
    try:
        from jfr.web.corpus_tasks import start_corpus_scheduler
        start_corpus_scheduler()
    except Exception as e:
        print(f"[corpus] scheduler init failed (manual refresh still works): {e}")
    yield


app = FastAPI(title="jfr", version="0.1.0", docs_url="/api/docs", lifespan=lifespan)

# Mount RAG router under /api/rag/* (search, query SSE, paper, graph, memory…)
try:
    from jfr.web.rag_routes import router as _rag_router
    app.include_router(_rag_router, prefix="/api/rag", tags=["rag"])
except Exception as e:
    print(f"[rag] router import failed: {e}")

_settings = get_settings()
_templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))
templates.env.filters["from_json"] = json.loads


def _conn():
    return get_conn(_settings.db_path)


def _days_since(iso_str: Optional[str]) -> int:
    if not iso_str:
        return 0
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).days
    except Exception:
        return 0


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


# ── Journals API ───────────────────────────────────────────────────────────────

@app.get("/api/journals")
def list_journals_api():
    conn = _conn()
    rows = conn.execute("SELECT * FROM journal ORDER BY name").fetchall()
    return [dict(r) for r in rows]


@app.get("/api/journals/{journal_id}/stats")
def journal_stats_api(journal_id: str):
    from jfr.corpus import corpus_stats
    conn = _conn()
    return corpus_stats(conn, journal_id)


# ── Add journal sources online (Crossref lookup) + manual add / delete ────────

@app.get("/api/journals/lookup")
def lookup_journals_online(q: str = ""):
    """Search Crossref's free journal directory by name or ISSN. Returns
    candidates the user can add — nothing is written to the DB here."""
    q = (q or "").strip()
    if not q:
        return {"results": []}
    import requests
    try:
        r = requests.get(
            "https://api.crossref.org/journals",
            params={"query": q, "rows": 10},
            headers={"User-Agent": "LabUI/1.0 (+journal-lookup; mailto:none@example.com)"},
            timeout=8,
        )
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", []) or []
    except Exception as e:
        raise HTTPException(502, f"Online lookup failed: {e}")

    results = []
    for it in items:
        issn_print = issn_elec = None
        for t in (it.get("issn-type") or []):
            if t.get("type") == "print":
                issn_print = t.get("value")
            elif t.get("type") == "electronic":
                issn_elec = t.get("value")
        issns = it.get("ISSN") or []
        if not issn_print and issns:
            issn_print = issns[0]
        name = it.get("title") or ""
        if not name:
            continue
        results.append({
            "name": name,
            "publisher": it.get("publisher") or "",
            "issn_print": issn_print,
            "issn_electronic": issn_elec,
        })
    return {"results": results}


class JournalAdd(BaseModel):
    name: str
    publisher: str = ""
    issn_print: Optional[str] = None
    issn_electronic: Optional[str] = None
    impact_factor: Optional[float] = None
    scope_statement: Optional[str] = None
    submission_url: Optional[str] = None
    is_hybrid_oa: bool = False
    is_fully_oa: bool = False
    abstract_format: str = "flat"
    id: Optional[str] = None


def _slug_journal_id(name: str) -> str:
    import re
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s[:48] or "journal"


@app.post("/api/journals", status_code=201)
def add_journal_api(body: JournalAdd):
    """Add a journal (from an online lookup pick or manual entry)."""
    from jfr.tracker import upsert_journal
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(400, "journal name is required")
    conn = _conn()
    existing = {r["id"] for r in conn.execute("SELECT id FROM journal").fetchall()}
    jid = (body.id or "").strip() or _slug_journal_id(name)
    base, n = jid, 2
    while jid in existing:
        jid = f"{base}_{n}"
        n += 1
    publisher = (body.publisher or "").strip() or "Unknown"
    upsert_journal(conn, {
        "id": jid, "name": name, "publisher": publisher,
        "publisher_family": publisher.lower().split()[0] if publisher else "unknown",
        "issn_print": body.issn_print, "issn_electronic": body.issn_electronic,
        "is_fully_oa": body.is_fully_oa, "is_hybrid_oa": body.is_hybrid_oa,
        "impact_factor": body.impact_factor,
        "submission_url": body.submission_url,
        "abstract_format": body.abstract_format or "flat",
        "scope_statement": body.scope_statement,
        "metadata": {},
    })
    return {"id": jid, "name": name}


@app.delete("/api/journals/{journal_id}")
def delete_journal_api(journal_id: str):
    conn = _conn()
    if not conn.execute("SELECT 1 FROM journal WHERE id=?", (journal_id,)).fetchone():
        raise HTTPException(404, "journal not found")
    # Refuse if a submission references it (would orphan tracked history).
    used = conn.execute("SELECT COUNT(*) AS c FROM submission WHERE journal_id=?", (journal_id,)).fetchone()
    if used and used["c"]:
        raise HTTPException(409, f"{used['c']} submission(s) reference this journal; cannot delete")
    conn.execute("DELETE FROM corpus_article WHERE journal_id=?", (journal_id,))
    conn.execute("DELETE FROM journal WHERE id=?", (journal_id,))
    conn.commit()
    return {"ok": True}


# ── Manuscripts API ────────────────────────────────────────────────────────────

@app.get("/api/manuscripts")
def list_manuscripts_api():
    from jfr.tracker import list_manuscripts as _list
    return _list(_conn())


@app.get("/api/manuscripts/{ms_id}")
def get_manuscript_api(ms_id: str):
    from jfr.tracker import get_manuscript as _get
    m = _get(_conn(), ms_id)
    if not m:
        raise HTTPException(404, f"Manuscript {ms_id!r} not found")
    return m


class ManuscriptCreate(BaseModel):
    title: str
    abstract: str
    principal_claim: str
    techniques: list[str] = []
    figures: list[str] = []
    bibtex_key: Optional[str] = None


@app.post("/api/manuscripts", status_code=201)
def create_manuscript_api(body: ManuscriptCreate):
    from jfr.tracker import create_manuscript as _create
    conn = _conn()
    ms_id = _create(
        conn, body.title, body.abstract, body.principal_claim,
        techniques=body.techniques, figures=body.figures, bibtex_key=body.bibtex_key,
    )
    return {"id": ms_id}


# ── Recommendations API ────────────────────────────────────────────────────────

@app.get("/api/recommend/{ms_id}")
def recommend_api(ms_id: str, top: int = Query(10, ge=1, le=30)):
    from jfr.tracker import get_manuscript
    from jfr.matching import ManuscriptInput, recommend as _recommend
    from jfr.web.corpus_tasks import open_vectors

    conn = _conn()
    ms = get_manuscript(conn, ms_id)
    if not ms:
        raise HTTPException(404, f"Manuscript {ms_id!r} not found")

    policy = load_policy(_settings.policy_toml)
    inp = ManuscriptInput(
        title=ms["title"],
        abstract=ms["abstract"],
        principal_claim=ms["principal_claim"],
        techniques=json.loads(ms["techniques_json"]),
        figures=json.loads(ms["figures_json"]),
    )
    with open_vectors(_settings) as qc:
        results = _recommend(
            inp, conn, qc, policy,
            _settings.abstract_model, _settings.claim_model,
            top_n=top, manuscript_id=ms_id,
        )
    return [r.to_dict() for r in results]


# ── Submissions API ────────────────────────────────────────────────────────────

@app.get("/api/submissions/active")
def active_submissions_api():
    from jfr.tracker import list_active_submissions
    return list_active_submissions(_conn())


@app.get("/api/submissions/{sub_id}")
def get_submission_api(sub_id: str):
    from jfr.tracker import get_submission as _get
    s = _get(_conn(), sub_id)
    if not s:
        raise HTTPException(404, f"Submission {sub_id!r} not found")
    return s


class SubmissionCreate(BaseModel):
    manuscript_id: str
    journal_id: str


@app.post("/api/submissions", status_code=201)
def create_submission_api(body: SubmissionCreate):
    from jfr.tracker import create_submission as _create
    conn = _conn()
    sub_id = _create(conn, body.manuscript_id, body.journal_id)
    return {"id": sub_id}


class TransitionRequest(BaseModel):
    to_state: str
    notes: Optional[str] = None


@app.post("/api/submissions/{sub_id}/transition")
def transition_submission_api(sub_id: str, body: TransitionRequest):
    from jfr.tracker import transition_submission as _transition, InvalidTransitionError
    conn = _conn()
    try:
        _transition(conn, sub_id, body.to_state, notes=body.notes)
    except InvalidTransitionError as e:
        raise HTTPException(422, str(e))
    return {"ok": True}


# ── Template: Dashboard ────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    from jfr.tracker import list_manuscripts as _list_ms, list_active_submissions
    conn = _conn()

    manuscripts = _list_ms(conn)
    active_raw = list_active_submissions(conn)
    active_subs = []
    for s in active_raw:
        d = dict(s)
        d["days_elapsed"] = _days_since(d.get("submitted_at") or d.get("created_at"))
        active_subs.append(d)

    total_subs = conn.execute("SELECT COUNT(*) FROM submission").fetchone()[0]
    journal_count = conn.execute("SELECT COUNT(*) FROM journal").fetchone()[0]

    # Cross-feature: pull RAG state for the at-a-glance view
    rag_paper_count = 0
    rag_chunk_count = 0
    chat_sessions: list[dict] = []
    chat_session_count = 0
    recent_findings: list[dict] = []
    recent_notes: list[dict] = []

    try:
        import os
        import sys
        from pathlib import Path
        _rag_dir = Path(
            os.environ.get("LABUI_RAG_DIR")
            or os.environ.get("CODEX_RAG_DIR")
            or str(Path(__file__).resolve().parents[3] / "Local_Rag" / "rag")
        )
        if str(_rag_dir) not in sys.path:
            sys.path.insert(0, str(_rag_dir))
        from config import DB_PATH as RAG_DB_PATH, NOTES_DIR
        if RAG_DB_PATH.exists():
            import sqlite3
            _rc = sqlite3.connect(str(RAG_DB_PATH))
            rag_paper_count = _rc.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            rag_chunk_count = _rc.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            _rc.close()
        try:
            import memory as _mem
            chat_sessions = _mem.get_sessions(limit=5)
            chat_session_count = len(_mem.get_sessions(limit=1000))
            recent_findings = _mem.get_all_memories(limit=5)
        except Exception:
            pass
        try:
            md_files = sorted(
                [p for p in NOTES_DIR.glob("*.md") if not p.name.startswith("._")],
                key=lambda x: x.stat().st_mtime, reverse=True,
            )[:3]
            for f in md_files:
                content = f.read_text(encoding="utf-8", errors="replace")
                title = f.stem
                for line in content.split("\n"):
                    if line.startswith("title:"):
                        title = line[6:].strip().strip('"\''); break
                    if line.startswith("# "):
                        title = line[2:].strip(); break
                recent_notes.append({
                    "id": f.stem, "title": title,
                    "modified": f.stat().st_mtime,
                    "preview": content[:140].replace("\n", " "),
                })
        except Exception:
            pass
    except Exception:
        pass

    return templates.TemplateResponse(request, "dashboard.html", {
        "manuscript_count": len(manuscripts),
        "active_subs": active_subs,
        "active_sub_count": len(active_subs),
        "total_subs": total_subs,
        "journal_count": journal_count,
        "manuscripts": manuscripts,
        "rag_paper_count": rag_paper_count,
        "rag_chunk_count": rag_chunk_count,
        "chat_sessions": chat_sessions,
        "chat_session_count": chat_session_count,
        "recent_findings": recent_findings,
        "recent_notes": recent_notes,
    })


# ── Template: Manuscripts ──────────────────────────────────────────────────────

@app.get("/manuscripts", response_class=HTMLResponse)
def manuscripts_page(request: Request):
    from jfr.tracker import list_manuscripts as _list
    return templates.TemplateResponse(request, "manuscripts.html", {
        "manuscripts": _list(_conn()),
    })


@app.get("/manuscripts/new", response_class=HTMLResponse)
def manuscripts_new_form(request: Request):
    return templates.TemplateResponse(request, "manuscript_new.html", {
        "error": None,
    })


@app.get("/manuscripts/{ms_id}/edit", response_class=HTMLResponse)
def manuscripts_edit_form(request: Request, ms_id: str):
    from jfr.tracker import get_manuscript
    ms = get_manuscript(_conn(), ms_id)
    if not ms:
        raise HTTPException(404, f"Manuscript {ms_id!r} not found")
    return templates.TemplateResponse(request, "manuscript_edit.html", {
        "ms": ms,
        "error": None,
    })


@app.post("/manuscripts/{ms_id}/edit", response_class=HTMLResponse)
async def manuscripts_edit_submit(
    request: Request,
    ms_id: str,
    title: str = Form(...),
    abstract: str = Form(...),
    principal_claim: str = Form(...),
    techniques: str = Form(""),
    bibtex_key: str = Form(""),
):
    from jfr.tracker import update_manuscript, get_manuscript
    conn = _conn()
    ms = get_manuscript(conn, ms_id)
    if not ms:
        raise HTTPException(404)
    tech_list = [t.strip() for t in techniques.split(",") if t.strip()]
    try:
        update_manuscript(
            conn, ms_id,
            title=title,
            abstract=abstract,
            principal_claim=principal_claim,
            techniques=tech_list,
            bibtex_key=bibtex_key or None,
        )
    except Exception as e:
        return templates.TemplateResponse(request, "manuscript_edit.html", {
            "ms": ms,
            "error": str(e),
        })
    return RedirectResponse(f"/manuscripts", status_code=303)


@app.post("/manuscripts/new", response_class=HTMLResponse)
async def manuscripts_new_submit(
    request: Request,
    ms_id: str = Form(""),
    title: str = Form(...),
    abstract: str = Form(...),
    principal_claim: str = Form(...),
    techniques: str = Form(""),
    bibtex_key: str = Form(""),
):
    from jfr.tracker import create_manuscript as _create
    conn = _conn()
    tech_list = [t.strip() for t in techniques.split(",") if t.strip()]
    try:
        created_id = _create(
            conn, title, abstract, principal_claim,
            techniques=tech_list,
            bibtex_key=bibtex_key or None,
            ms_id=ms_id or None,
        )
    except Exception as e:
        return templates.TemplateResponse(request, "manuscript_new.html", {
            "error": str(e),
        })
    return RedirectResponse(f"/recommend?ms={created_id}", status_code=303)


# ── Template: Recommend ────────────────────────────────────────────────────────

@app.get("/recommend", response_class=HTMLResponse)
def recommend_page(request: Request, ms: str = ""):
    from jfr.tracker import list_manuscripts as _list, get_manuscript
    conn = _conn()
    manuscripts = _list(conn)
    selected_ms = get_manuscript(conn, ms) if ms else None
    policy = load_policy(_settings.policy_toml)

    return templates.TemplateResponse(request, "recommend.html", {
        "manuscripts": manuscripts,
        "selected_ms": selected_ms,
        "results": None,
        "policy": policy,
    })


@app.get("/htmx/recommend/{ms_id}", response_class=HTMLResponse)
def htmx_recommend(request: Request, ms_id: str, top: int = Query(10, ge=1, le=15)):
    from jfr.tracker import get_manuscript
    from jfr.matching import ManuscriptInput, recommend as _recommend
    from jfr.web.corpus_tasks import open_vectors

    conn = _conn()
    ms = get_manuscript(conn, ms_id)
    if not ms:
        return HTMLResponse(
            "<div class='text-red-600 p-6'>Manuscript not found</div>", status_code=404
        )

    policy = load_policy(_settings.policy_toml)
    inp = ManuscriptInput(
        title=ms["title"],
        abstract=ms["abstract"],
        principal_claim=ms["principal_claim"],
        techniques=json.loads(ms["techniques_json"]),
        figures=json.loads(ms["figures_json"]),
    )
    with open_vectors(_settings) as qc:
        results = _recommend(
            inp, conn, qc, policy,
            _settings.abstract_model, _settings.claim_model,
            top_n=top, manuscript_id=ms_id,
        )
    result_dicts = [r.to_dict() for r in results]

    return templates.TemplateResponse(request, "_results_partial.html", {
        "results": result_dicts,
        "ms_id": ms_id,
    })


# ── Template: Submissions ──────────────────────────────────────────────────────

@app.get("/submissions", response_class=HTMLResponse)
def submissions_page(request: Request):
    from jfr.tracker import list_active_submissions
    conn = _conn()

    active_raw = list_active_submissions(conn)
    active = []
    for s in active_raw:
        d = dict(s)
        d["days_elapsed"] = _days_since(d.get("submitted_at") or d.get("created_at"))
        active.append(d)

    all_subs = conn.execute(
        """SELECT s.*, m.title as manuscript_title, j.name as journal_name
           FROM submission s
           JOIN manuscript m ON m.id = s.manuscript_id
           JOIN journal j ON j.id = s.journal_id
           ORDER BY s.created_at DESC"""
    ).fetchall()

    return templates.TemplateResponse(request, "submissions.html", {
        "active": active,
        "all_subs": [dict(r) for r in all_subs],
    })


@app.post("/submissions/create", response_class=HTMLResponse)
async def submissions_create(
    manuscript_id: str = Form(...),
    journal_id: str = Form(...),
):
    from jfr.tracker import create_submission as _create
    conn = _conn()
    sub_id = _create(conn, manuscript_id, journal_id)
    return RedirectResponse(f"/submissions/{sub_id}", status_code=303)


@app.get("/submissions/{sub_id}", response_class=HTMLResponse)
def submission_detail(request: Request, sub_id: str):
    from jfr.tracker import get_submission as _get
    conn = _conn()

    sub = _get(conn, sub_id)
    if not sub:
        raise HTTPException(404, f"Submission {sub_id!r} not found")

    ms_row = conn.execute("SELECT * FROM manuscript WHERE id=?", (sub["manuscript_id"],)).fetchone()
    journal_row = conn.execute("SELECT * FROM journal WHERE id=?", (sub["journal_id"],)).fetchone()
    journal = dict(journal_row) if journal_row else {}
    manuscript = dict(ms_row) if ms_row else {}

    sub["manuscript_title"] = manuscript.get("title", sub["manuscript_id"])
    sub["journal_name"] = journal.get("name", sub["journal_id"])

    allowed = VALID_TRANSITIONS.get(sub["current_state"], [])
    days_elapsed = _days_since(sub.get("submitted_at") or sub.get("created_at"))

    return templates.TemplateResponse(request, "submission_detail.html", {
        "sub": sub,
        "journal": journal,
        "manuscript": manuscript,
        "allowed_transitions": allowed,
        "days_elapsed": days_elapsed,
    })


@app.post("/submissions/{sub_id}/transition", response_class=HTMLResponse)
async def submission_transition_form(
    sub_id: str,
    to_state: str = Form(...),
    notes: str = Form(""),
):
    from jfr.tracker import transition_submission as _transition, InvalidTransitionError
    conn = _conn()
    try:
        _transition(conn, sub_id, to_state, notes=notes or None)
    except InvalidTransitionError as e:
        raise HTTPException(422, str(e))
    return RedirectResponse(f"/submissions/{sub_id}", status_code=303)


@app.post("/submissions/{sub_id}/comments", response_class=HTMLResponse)
async def submission_add_comment(
    sub_id: str,
    review_round: int = Form(1),
    reviewer_number: int = Form(1),
    comment_text: str = Form(...),
):
    from jfr.tracker import add_reviewer_comment
    conn = _conn()
    add_reviewer_comment(conn, sub_id, reviewer_number, comment_text, round=review_round)
    return RedirectResponse(f"/submissions/{sub_id}", status_code=303)


@app.post("/submissions/{sub_id}/comments/{comment_id}/edit", response_class=HTMLResponse)
async def submission_edit_comment(
    sub_id: str,
    comment_id: int,
    review_round: int = Form(1),
    reviewer_number: int = Form(1),
    comment_text: str = Form(...),
    response_text: str = Form(""),
    received_at: str = Form(""),
):
    from jfr.tracker import update_reviewer_comment
    conn = _conn()
    update_reviewer_comment(
        conn, comment_id,
        review_round=review_round,
        reviewer_number=reviewer_number,
        comment_text=comment_text,
        response_text=response_text or None,
        received_at=received_at or None,
    )
    return RedirectResponse(f"/submissions/{sub_id}", status_code=303)


# ── Template: Journals ─────────────────────────────────────────────────────────

@app.get("/journals", response_class=HTMLResponse)
def journals_page(request: Request):
    conn = _conn()
    journals_raw = conn.execute("SELECT * FROM journal ORDER BY name").fetchall()
    journals = []
    for j in journals_raw:
        d = dict(j)
        counts = conn.execute(
            """SELECT
                 COUNT(*) as article_count,
                 SUM(CASE WHEN embedding_model IS NOT NULL THEN 1 ELSE 0 END) as embedded_count
               FROM corpus_article WHERE journal_id=?""",
            (j["id"],),
        ).fetchone()
        d["article_count"] = counts["article_count"]
        d["embedded_count"] = counts["embedded_count"] or 0
        journals.append(d)

    from jfr.web.corpus_tasks import get_status as _corpus_status
    return templates.TemplateResponse(request, "journals.html", {
        "journals": journals,
        "corpus_status": _corpus_status(),
    })


# ── API: corpus refresh (fetch recent articles + embed) ──────────────────────

@app.get("/api/corpus/refresh/status")
def corpus_refresh_status_api():
    """Live status of the background corpus refresh (for UI polling)."""
    from jfr.web.corpus_tasks import get_status
    return get_status()


@app.post("/api/corpus/refresh/run")
def corpus_refresh_run_api(journal_id: Optional[str] = Query(None)):
    """Kick a one-shot corpus refresh in the background. Pass ?journal_id= to
    refresh a single journal, otherwise all journals are refreshed."""
    from jfr.web.corpus_tasks import trigger_refresh, get_status
    started = trigger_refresh(journal_ids=[journal_id] if journal_id else None)
    status = get_status()
    if not started:
        return {"started": False, "message": "A refresh is already running.", "status": status}
    return {"started": True, "message": "Corpus refresh started.", "status": status}


# ── API: RAG ↔ JFR cross-feature (manuscript context, paper links, lab view) ──

@app.get("/api/rag/manuscript/{ms_id}")
def rag_manuscript_context_api(ms_id: str):
    """Get RAG context for a manuscript (linked papers + search)."""
    try:
        from integration.rag_integration import get_manuscript_context
        return get_manuscript_context(ms_id)
    except Exception as e:
        return {"error": f"Failed to get manuscript context: {e}", "manuscript": None}


@app.post("/api/rag/links")
def rag_link_paper(ms_id: str = Form(...),
                   paper_id: str = Form(...),
                   link_type: str = Form("related"),
                   note: str = Form("")):
    """Link a RAG paper to a manuscript."""
    try:
        from integration.linking import link_manuscript_to_paper
        ok = link_manuscript_to_paper(ms_id, paper_id, link_type, note)
        if ok:
            return {"success": True, "message": "Paper link created"}
        raise HTTPException(500, "Link creation failed")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Link creation failed: {e}")


@app.get("/api/rag/links/{ms_id}")
def rag_get_links(ms_id: str):
    """Get all paper links for a manuscript."""
    try:
        from integration.linking import get_linked_papers
        links = get_linked_papers(ms_id)
        return {"links": links, "total": len(links)}
    except Exception as e:
        return {"error": f"Failed to get links: {e}", "links": []}


@app.delete("/api/rag/links/{link_id}")
def rag_delete_link(link_id: int):
    """Unlink a paper from a manuscript."""
    try:
        from integration.linking import delete_link
        ok = delete_link(link_id)
        if not ok:
            raise HTTPException(404, "Link not found")
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to delete link: {e}")


@app.get("/api/rag/lab/{ms_id}")
def rag_lab_dashboard_api(ms_id: str):
    """Get research lab dashboard data for a manuscript."""
    try:
        from integration.lab_view import research_lab_dashboard
        return research_lab_dashboard(ms_id)
    except Exception as e:
        return {"error": f"Failed to get lab dashboard: {e}", "template_data": {}}


@app.get("/api/rag/lab/stats")
def rag_lab_stats_api():
    """Get research lab statistics across all manuscripts."""
    try:
        from integration.lab_view import get_research_lab_stats
        return get_research_lab_stats()
    except Exception as e:
        return {"error": f"Failed to get stats: {e}", "stats": {}}


# ── Template: Research Lab ─────────────────────────────────

@app.get("/lab", response_class=HTMLResponse)
def research_lab_page(request: Request, ms_id: str = ""):
    """Research lab page combining RAG and JFR."""
    conn = _conn()
    
    # Get manuscripts for selector
    manuscripts = conn.execute(
        "SELECT id, title FROM manuscript ORDER BY created_at DESC"
    ).fetchall()
    
    # Get lab stats
    try:
        from integration.lab_view import get_research_lab_stats
        stats = get_research_lab_stats()
    except Exception:
        stats = {}
    
    return templates.TemplateResponse(request, "research_lab.html", {
        "manuscripts": [dict(m) for m in manuscripts],
        "selected_ms": ms_id,
        "stats": stats,
    })


# ── Templates: Research Lab pages (unified UI for /api/rag/*) ─────────────────

@app.get("/research/chat", response_class=HTMLResponse)
def research_chat_page(request: Request):
    return templates.TemplateResponse(request, "research_chat.html", {})


@app.get("/research/search", response_class=HTMLResponse)
def research_search_page(request: Request):
    return templates.TemplateResponse(request, "research_search.html", {})


@app.get("/research/memory", response_class=HTMLResponse)
def research_memory_page(request: Request):
    return templates.TemplateResponse(request, "research_memory.html", {})


@app.get("/research/notes", response_class=HTMLResponse)
def research_notes_page(request: Request):
    return templates.TemplateResponse(request, "research_notes.html", {})


@app.get("/research/style", response_class=HTMLResponse)
def research_style_page(request: Request):
    return templates.TemplateResponse(request, "research_style.html", {})


@app.get("/research/graph", response_class=HTMLResponse)
def research_graph_page(request: Request):
    return templates.TemplateResponse(request, "research_graph.html", {})


@app.get("/research/upload", response_class=HTMLResponse)
def research_upload_page(request: Request):
    return templates.TemplateResponse(request, "research_upload.html", {})


@app.get("/research/paper/{paper_id}/view", response_class=HTMLResponse)
def research_paper_view_page(request: Request, paper_id: str):
    return templates.TemplateResponse(request, "research_paper_view.html", {"paper_id": paper_id})


# ── Experiments API ──────────────────────────────────────────────────────────

from jfr.db.schema import EXPERIMENT_STATUSES  # noqa: E402

class ExperimentBody(BaseModel):
    manuscript_id: Optional[str] = None
    name: str
    ran_on: Optional[str] = None        # YYYY-MM-DD
    scheduled_for: Optional[str] = None # YYYY-MM-DD
    status: str = "planned"
    objective: Optional[str] = None
    methodology: Optional[str] = None
    conditions_json: Optional[str] = "{}"
    equipment: Optional[str] = None
    observations: Optional[str] = None
    results_md: Optional[str] = None
    notes_md: Optional[str] = None
    tags_json: Optional[str] = "[]"
    linked_papers_json: Optional[str] = "[]"


def _ms_title_map(conn) -> dict[str, str]:
    rows = conn.execute("SELECT id, title FROM manuscript").fetchall()
    return {r["id"]: r["title"] for r in rows}


def _experiment_row_to_dict(row, ms_titles: dict[str, str]) -> dict:
    d = dict(row)
    d["manuscript_title"] = ms_titles.get(d.get("manuscript_id"), None)
    try:    d["tags"] = json.loads(d.get("tags_json") or "[]")
    except: d["tags"] = []
    try:    d["conditions"] = json.loads(d.get("conditions_json") or "{}")
    except: d["conditions"] = {}
    try:    d["linked_papers"] = json.loads(d.get("linked_papers_json") or "[]")
    except: d["linked_papers"] = []
    return d


@app.get("/api/experiments")
def list_experiments_api(manuscript_id: Optional[str] = None, status: Optional[str] = None):
    conn = _conn()
    sql = "SELECT * FROM experiment WHERE 1=1"
    params: list = []
    if manuscript_id:
        sql += " AND manuscript_id = ?"; params.append(manuscript_id)
    if status:
        sql += " AND status = ?"; params.append(status)
    sql += " ORDER BY COALESCE(ran_on, scheduled_for, created_at) DESC"
    rows = conn.execute(sql, params).fetchall()
    titles = _ms_title_map(conn)
    return [_experiment_row_to_dict(r, titles) for r in rows]


@app.get("/api/experiments/upcoming")
def upcoming_experiments_api():
    conn = _conn()
    today = datetime.now().strftime("%Y-%m-%d")
    rows = conn.execute(
        "SELECT * FROM experiment"
        " WHERE status = 'planned' AND scheduled_for IS NOT NULL AND scheduled_for >= ?"
        " ORDER BY scheduled_for ASC",
        (today,),
    ).fetchall()
    titles = _ms_title_map(conn)
    return [_experiment_row_to_dict(r, titles) for r in rows]


@app.get("/api/experiments/{exp_id}")
def get_experiment_api(exp_id: str):
    conn = _conn()
    row = conn.execute("SELECT * FROM experiment WHERE id=?", (exp_id,)).fetchone()
    if not row:
        raise HTTPException(404, "experiment not found")
    titles = _ms_title_map(conn)
    return _experiment_row_to_dict(row, titles)


def _next_exp_id(conn) -> str:
    """Sequential id: exp_001, exp_002, …"""
    row = conn.execute("SELECT id FROM experiment ORDER BY rowid DESC LIMIT 1").fetchone()
    if not row:
        return "exp_001"
    last = row["id"]
    if last.startswith("exp_") and last[4:].isdigit():
        return f"exp_{int(last[4:]) + 1:03d}"
    return f"exp_{datetime.now().strftime('%Y%m%d%H%M%S')}"


@app.post("/api/experiments", status_code=201)
def create_experiment_api(body: ExperimentBody):
    if body.status not in EXPERIMENT_STATUSES:
        raise HTTPException(422, f"invalid status: {body.status!r}")
    conn = _conn()
    exp_id = _next_exp_id(conn)
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        """INSERT INTO experiment
           (id, manuscript_id, name, ran_on, scheduled_for, status,
            objective, methodology, conditions_json, equipment, observations,
            results_md, notes_md, tags_json, linked_papers_json,
            created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (exp_id, body.manuscript_id, body.name, body.ran_on, body.scheduled_for,
         body.status, body.objective, body.methodology, body.conditions_json or "{}",
         body.equipment, body.observations, body.results_md, body.notes_md,
         body.tags_json or "[]", body.linked_papers_json or "[]",
         now, now),
    )
    conn.commit()
    return {"id": exp_id}


@app.put("/api/experiments/{exp_id}")
def update_experiment_api(exp_id: str, body: ExperimentBody):
    if body.status not in EXPERIMENT_STATUSES:
        raise HTTPException(422, f"invalid status: {body.status!r}")
    conn = _conn()
    row = conn.execute("SELECT id FROM experiment WHERE id=?", (exp_id,)).fetchone()
    if not row:
        raise HTTPException(404, "experiment not found")
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        """UPDATE experiment SET
              manuscript_id=?, name=?, ran_on=?, scheduled_for=?, status=?,
              objective=?, methodology=?, conditions_json=?, equipment=?, observations=?,
              results_md=?, notes_md=?, tags_json=?, linked_papers_json=?, updated_at=?
           WHERE id=?""",
        (body.manuscript_id, body.name, body.ran_on, body.scheduled_for, body.status,
         body.objective, body.methodology, body.conditions_json or "{}",
         body.equipment, body.observations, body.results_md, body.notes_md,
         body.tags_json or "[]", body.linked_papers_json or "[]",
         now, exp_id),
    )
    conn.commit()
    return {"ok": True}


@app.delete("/api/experiments/{exp_id}")
def delete_experiment_api(exp_id: str):
    conn = _conn()
    res = conn.execute("DELETE FROM experiment WHERE id=?", (exp_id,))
    conn.commit()
    if res.rowcount == 0:
        raise HTTPException(404, "experiment not found")
    return {"ok": True}


# ── Experiments: page routes ─────────────────────────────────────────────────

@app.get("/experiments", response_class=HTMLResponse)
def experiments_page(request: Request):
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM experiment ORDER BY COALESCE(ran_on, scheduled_for, created_at) DESC"
    ).fetchall()
    titles = _ms_title_map(conn)
    experiments = [_experiment_row_to_dict(r, titles) for r in rows]

    # Group: by manuscript_id (titles for headers), plus "unassigned"
    groups: dict[str, dict] = {}
    for e in experiments:
        mid = e.get("manuscript_id") or ""
        if mid not in groups:
            groups[mid] = {
                "manuscript_id": mid or None,
                "manuscript_title": titles.get(mid, "Unassigned") if mid else "Unassigned",
                "experiments": [],
            }
        groups[mid]["experiments"].append(e)

    # Order: assigned groups first (by title), then unassigned last
    assigned = sorted(
        (g for k, g in groups.items() if k),
        key=lambda g: (g["manuscript_title"] or "").lower(),
    )
    unassigned = groups.get("", None)
    grouped = assigned + ([unassigned] if unassigned else [])

    # Upcoming: planned + future scheduled_for
    today = datetime.now().strftime("%Y-%m-%d")
    upcoming = [
        e for e in experiments
        if e.get("status") == "planned" and (e.get("scheduled_for") or "") >= today
    ]
    upcoming.sort(key=lambda e: e.get("scheduled_for") or "")

    manuscripts = conn.execute("SELECT id, title FROM manuscript ORDER BY title").fetchall()

    return templates.TemplateResponse(request, "experiments.html", {
        "grouped": grouped,
        "upcoming": upcoming,
        "manuscripts": [dict(m) for m in manuscripts],
        "total_count": len(experiments),
        "statuses": EXPERIMENT_STATUSES,
    })


@app.get("/experiments/new", response_class=HTMLResponse)
def experiment_new_page(request: Request, ms: str = ""):
    conn = _conn()
    manuscripts = conn.execute("SELECT id, title FROM manuscript ORDER BY title").fetchall()
    return templates.TemplateResponse(request, "experiment_form.html", {
        "mode": "new",
        "exp": None,
        "preselect_ms": ms,
        "manuscripts": [dict(m) for m in manuscripts],
        "statuses": EXPERIMENT_STATUSES,
    })


@app.get("/experiments/{exp_id}", response_class=HTMLResponse)
def experiment_detail_page(request: Request, exp_id: str):
    conn = _conn()
    row = conn.execute("SELECT * FROM experiment WHERE id=?", (exp_id,)).fetchone()
    if not row:
        raise HTTPException(404, f"Experiment {exp_id!r} not found")
    titles = _ms_title_map(conn)
    exp = _experiment_row_to_dict(row, titles)
    return templates.TemplateResponse(request, "experiment_detail.html", {"exp": exp})


@app.get("/experiments/{exp_id}/edit", response_class=HTMLResponse)
def experiment_edit_page(request: Request, exp_id: str):
    conn = _conn()
    row = conn.execute("SELECT * FROM experiment WHERE id=?", (exp_id,)).fetchone()
    if not row:
        raise HTTPException(404, f"Experiment {exp_id!r} not found")
    titles = _ms_title_map(conn)
    exp = _experiment_row_to_dict(row, titles)
    manuscripts = conn.execute("SELECT id, title FROM manuscript ORDER BY title").fetchall()
    return templates.TemplateResponse(request, "experiment_form.html", {
        "mode": "edit",
        "exp": exp,
        "preselect_ms": exp.get("manuscript_id") or "",
        "manuscripts": [dict(m) for m in manuscripts],
        "statuses": EXPERIMENT_STATUSES,
    })


@app.get("/schedule", response_class=HTMLResponse)
def schedule_page(request: Request, month: Optional[str] = None):
    """month param: YYYY-MM. Defaults to current month."""
    if not month:
        month = datetime.now().strftime("%Y-%m")
    try:
        y, m = month.split("-"); y, m = int(y), int(m)
    except Exception:
        raise HTTPException(400, f"bad month {month!r}; expected YYYY-MM")

    conn = _conn()
    # First + last day of month
    from calendar import monthrange
    last_day = monthrange(y, m)[1]
    first_iso = f"{y:04d}-{m:02d}-01"
    last_iso  = f"{y:04d}-{m:02d}-{last_day:02d}"
    rows = conn.execute(
        "SELECT * FROM experiment"
        " WHERE scheduled_for IS NOT NULL"
        "   AND scheduled_for BETWEEN ? AND ?"
        " ORDER BY scheduled_for ASC",
        (first_iso, last_iso),
    ).fetchall()
    titles = _ms_title_map(conn)
    expmts = [_experiment_row_to_dict(r, titles) for r in rows]

    # Bucket by ISO date
    by_day: dict[str, list] = {}
    for e in expmts:
        by_day.setdefault(e.get("scheduled_for"), []).append(e)

    # Calendar grid: weeks of (date|None) cells starting on Monday
    from datetime import date, timedelta
    first = date(y, m, 1)
    # back up to Monday of first week
    grid_start = first - timedelta(days=first.weekday())
    weeks = []
    cur = grid_start
    while True:
        week = []
        for _ in range(7):
            week.append({
                "iso": cur.strftime("%Y-%m-%d"),
                "day": cur.day,
                "in_month": (cur.month == m),
                "is_today": cur == datetime.now().date(),
                "experiments": by_day.get(cur.strftime("%Y-%m-%d"), []),
            })
            cur += timedelta(days=1)
        weeks.append(week)
        if cur.month != m and cur > date(y, m, last_day):
            break

    # Prev/next month
    prev_y, prev_m = (y - 1, 12) if m == 1 else (y, m - 1)
    next_y, next_m = (y + 1, 1) if m == 12 else (y, m + 1)
    label = first.strftime("%B %Y")

    return templates.TemplateResponse(request, "schedule.html", {
        "weeks": weeks,
        "month": month,
        "month_label": label,
        "prev_month": f"{prev_y:04d}-{prev_m:02d}",
        "next_month": f"{next_y:04d}-{next_m:02d}",
        "experiment_count": len(expmts),
    })
