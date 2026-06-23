"""Background corpus refresh for the Journal-Fit Recommender.

Fetches recent articles per journal from OpenAlex (or CrossRef), ingests the new
ones into the SQLite corpus, and embeds any missing vectors into Qdrant — on a
weekly schedule and on demand from the web UI.

Mirrors the RAG auto-indexer pattern (daemon + status dict + manual trigger),
adapted for the journal recommender's SQLite corpus + local Qdrant store.

Both fetch and embed are incremental: `ingest_articles` skips DOIs already
present and `embed_corpus` only embeds articles without a current vector, so
every run after the first is cheap.

Configuration (env vars, all optional):
  JFR_CORPUS_REFRESH_DAYS   interval in days for the scheduled run (default 7;
                            set <= 0 to disable scheduling, button still works)
  JFR_CORPUS_AUTOSTART      "0"/"false"/"no" to skip the empty-corpus populate
                            on startup (default on)
  JFR_CORPUS_SOURCE         "openalex" (default) or "crossref"
  JFR_CORPUS_MONTHS         corpus window in months (default 36)
  JFR_CORPUS_LIMIT          max articles fetched per journal (default 500)
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from jfr.config import get_settings
from jfr.db import get_conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_flag(name: str, default: bool = True) -> bool:
    return os.environ.get(name, "1" if default else "0").lower() not in ("0", "false", "no")


REFRESH_DAYS = int(os.environ.get("JFR_CORPUS_REFRESH_DAYS", "7"))
SOURCE = os.environ.get("JFR_CORPUS_SOURCE", "openalex")
MONTHS = int(os.environ.get("JFR_CORPUS_MONTHS", "36"))
LIMIT = int(os.environ.get("JFR_CORPUS_LIMIT", "500"))


# ── Shared Qdrant access ─────────────────────────────────────────────────────
# The embedded (file-backed) Qdrant backend holds an OS-level file lock and
# allows only ONE client at a time. The recommend endpoints and this background
# embedder must therefore take turns. Every place in the web process that opens a
# QdrantClient(path=…) MUST go through open_vectors() so they share this lock.
_vectors_lock = threading.RLock()


@contextmanager
def open_vectors(settings=None):
    """Yield a QdrantClient on the local vectors dir, serialised process-wide.

    Acquires the shared lock, opens the client, and guarantees both the client
    is closed (releasing the file lock) and the in-process lock is released, even
    on error."""
    from qdrant_client import QdrantClient

    s = settings or get_settings()
    with _vectors_lock:
        qc = QdrantClient(path=str(s.vectors_dir))
        try:
            yield qc
        finally:
            qc.close()


# ── Refresh state + execution ────────────────────────────────────────────────
_state: dict = {
    "running": False,
    "phase": "idle",            # idle | fetching | embedding | done | error
    "message": "Idle",
    "journal": None,
    "journals_done": 0,
    "journals_total": 0,
    "articles_added": 0,
    "vectors_added": 0,
    "started_at": None,
    "finished_at": None,
    "last_run": None,
    "next_run": None,
    "error": None,
}
_state_lock = threading.Lock()
_run_lock = threading.Lock()    # guarantees a single refresh runs at a time
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="corpus-refresh")
_scheduler = None


def _set(**kw) -> None:
    with _state_lock:
        _state.update(**kw)


def get_status() -> dict:
    """Snapshot of the current refresh state (safe to serialise as JSON)."""
    with _state_lock:
        st = dict(_state)
    # next_run is owned by the scheduler — read it live so it stays accurate.
    if _scheduler is not None:
        try:
            job = _scheduler.get_job("corpus_refresh")
            st["next_run"] = job.next_run_time.isoformat() if job and job.next_run_time else None
        except Exception:
            pass
    st["scheduled"] = _scheduler is not None
    return st


def _refresh_worker(
    journal_ids: Optional[list[str]] = None,
    months: Optional[int] = None,
    limit: Optional[int] = None,
    source: Optional[str] = None,
) -> None:
    """Run one full refresh pass. Held to a single concurrent execution by
    _run_lock; extra triggers while running are no-ops."""
    if not _run_lock.acquire(blocking=False):
        return  # a refresh is already in progress

    from jfr.corpus import (
        fetch_openalex_articles,
        fetch_crossref_articles,
        ingest_articles,
        embed_corpus,
    )
    from qdrant_client.models import Distance, VectorParams

    s = get_settings()
    months = months or MONTHS
    limit = limit or LIMIT
    source = source or SOURCE
    conn = get_conn(s.db_path)
    try:
        if journal_ids:
            rows = [
                conn.execute("SELECT * FROM journal WHERE id=?", (jid,)).fetchone()
                for jid in journal_ids
            ]
            journals = [dict(r) for r in rows if r]
        else:
            journals = [dict(r) for r in conn.execute("SELECT * FROM journal ORDER BY id").fetchall()]

        _set(
            running=True, phase="fetching", error=None, journal=None,
            journals_total=len(journals), journals_done=0,
            articles_added=0, vectors_added=0,
            started_at=_now(), finished_at=None,
            message=f"Refreshing {len(journals)} journal(s)…",
        )

        total_added = 0
        total_vecs = 0
        for j in journals:
            jid = j["id"]
            issn = j.get("issn_electronic") or j.get("issn_print")

            # 1) Fetch + ingest (network only — no Qdrant lock needed).
            if issn:
                _set(journal=jid, phase="fetching", message=f"Fetching {jid}…")
                try:
                    if source == "crossref":
                        articles = fetch_crossref_articles(issn, months=months, limit=limit)
                    else:
                        articles = fetch_openalex_articles(issn, months=months, limit=limit)
                    total_added += ingest_articles(conn, jid, articles)
                except Exception as e:
                    _set(message=f"{jid}: fetch error: {e}")
            else:
                _set(journal=jid, message=f"{jid}: no ISSN, skipping fetch")

            # 2) Embed missing vectors. Take the shared Qdrant lock per journal so
            #    recommend requests can slip in between journals.
            _set(phase="embedding", message=f"Embedding {jid}…", articles_added=total_added)
            try:
                collection = f"journal_{jid}"
                with open_vectors(s) as qc:
                    try:
                        qc.get_collection(collection)
                    except Exception:
                        qc.create_collection(
                            collection_name=collection,
                            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                        )
                    total_vecs += embed_corpus(conn, jid, qc, collection, s.abstract_model)
            except Exception as e:
                _set(message=f"{jid}: embed error: {e}")

            with _state_lock:
                _state["journals_done"] += 1
                _state["vectors_added"] = total_vecs
                _state["articles_added"] = total_added

        _set(
            running=False, phase="done", journal=None,
            finished_at=_now(), last_run=_now(),
            message=f"Done — {total_added} new article(s), {total_vecs} vector(s) embedded.",
        )
    except Exception as e:
        _set(running=False, phase="error", journal=None, error=str(e),
             message=f"Refresh failed: {e}")
    finally:
        conn.close()
        _run_lock.release()


def trigger_refresh(
    journal_ids: Optional[list[str]] = None,
    months: Optional[int] = None,
    limit: Optional[int] = None,
    source: Optional[str] = None,
) -> bool:
    """Kick a one-shot refresh in the background. Returns False if one is already
    running (the caller can surface that to the user)."""
    if _state.get("running"):
        return False
    _executor.submit(_refresh_worker, journal_ids, months, limit, source)
    return True


def _corpus_is_empty() -> bool:
    s = get_settings()
    conn = get_conn(s.db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM corpus_article").fetchone()[0] == 0
    finally:
        conn.close()


def start_corpus_scheduler() -> None:
    """Start the weekly scheduled refresh and, if the corpus is empty, kick an
    immediate background populate. Safe to call once at app startup."""
    global _scheduler

    # Auto-populate an empty corpus on startup so a fresh install yields real
    # recommendations without any manual CLI step.
    if _env_flag("JFR_CORPUS_AUTOSTART", True):
        try:
            if _corpus_is_empty():
                _set(message="Corpus empty — starting initial populate in background…")
                trigger_refresh()
                print("[corpus] empty corpus detected — initial populate started")
        except Exception as e:
            print(f"[corpus] empty-corpus check failed: {e}")

    if REFRESH_DAYS <= 0:
        print("[corpus] scheduled refresh disabled (JFR_CORPUS_REFRESH_DAYS<=0)")
        return

    try:
        from apscheduler.schedulers.background import BackgroundScheduler

        _scheduler = BackgroundScheduler(daemon=True)
        _scheduler.add_job(
            lambda: trigger_refresh(),
            trigger="interval",
            days=REFRESH_DAYS,
            id="corpus_refresh",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        _scheduler.start()
        print(f"[corpus] scheduled refresh every {REFRESH_DAYS} day(s)")
    except Exception as e:
        _scheduler = None
        print(f"[corpus] scheduler start failed (manual refresh still works): {e}")
