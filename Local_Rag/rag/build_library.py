#!/usr/bin/env python3
"""
One-shot, leave-it-running library builder.

Runs the full pipeline over the entire RAG library, in order:
  1. Parse PDFs            — OCR fallback for broken-font (garbled) PDFs
  2. Build index           — chunk + embed any new/repaired papers
  3. Repair garbled papers — re-OCR + reindex anything already indexed as glyph soup
  4. Summaries             — generate a clear prose summary for every paper missing one

Each phase is incremental and safe to re-run. Per-paper errors are logged and
skipped so one bad file never stops the whole job.

Run:
    python3 build_library.py                 # full pipeline
    python3 build_library.py --force-summaries   # also regenerate existing summaries
    python3 build_library.py --skip-reprocess    # skip the re-OCR repair phase
    python3 build_library.py --only-summaries    # just (re)generate summaries
"""
import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from config import DB_PATH, OLLAMA_BASE_URL


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.enable_load_extension(True)
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
    except Exception:
        pass
    conn.enable_load_extension(False)
    return conn


def phase_parse_index() -> None:
    from ingest import parse_pdfs, build_index
    _log("Phase 1/4 — parsing PDFs (OCR fallback for garbled fonts)…")
    parse_pdfs.run()
    _log("Phase 2/4 — building index (chunk + embed new papers)…")
    build_index.run()


def phase_reprocess() -> None:
    from ingest import reprocess_garbled
    _log("Phase 3/4 — repairing garbled papers (re-OCR + reindex)…")
    result = reprocess_garbled.run()
    _log(f"  repaired {result.get('repaired')} of {result.get('found')} garbled paper(s)")


def phase_summaries(force: bool = False) -> None:
    from generation.summarize import resolve_gen_model, summarize_paper, ensure_summary_col

    _log("Phase 4/4 — generating summaries…")
    if not DB_PATH.exists():
        _log("  no index database yet — nothing to summarize.")
        return

    model = resolve_gen_model(OLLAMA_BASE_URL)
    if not model:
        _log("  ERROR: no Ollama model available (is Ollama running? is a model pulled?). "
             "Skipping summaries.")
        return
    _log(f"  using model: {model}")

    conn = _open_db()
    ensure_summary_col(conn)
    if force:
        rows = conn.execute("SELECT paper_id FROM papers").fetchall()
    else:
        rows = conn.execute(
            "SELECT paper_id FROM papers WHERE summary IS NULL OR summary = ''"
        ).fetchall()
    ids = [r[0] for r in rows]
    total = len(ids)
    _log(f"  {total} paper(s) to summarize")

    done = ok = 0
    for pid in ids:
        t0 = time.time()
        try:
            summary = summarize_paper(conn, pid, model, OLLAMA_BASE_URL, force=force)
            if summary:
                ok += 1
                status = "ok"
            else:
                status = "skipped (insufficient/garbled text)"
        except Exception as e:
            status = f"error: {e}"
        done += 1
        _log(f"  [{done}/{total}] {pid[:48]:<48} {status}  ({time.time()-t0:.0f}s)")
    conn.close()
    _log(f"  summaries complete: {ok}/{total} generated")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the whole RAG library end-to-end.")
    ap.add_argument("--force-summaries", action="store_true",
                    help="Regenerate ALL summaries, even ones that already exist")
    ap.add_argument("--skip-reprocess", action="store_true",
                    help="Skip the garbled-paper re-OCR repair phase")
    ap.add_argument("--only-summaries", action="store_true",
                    help="Only run the summary phase (skip parse/index/repair)")
    args = ap.parse_args()

    t0 = time.time()
    _log("=== LabUI library build started ===")
    try:
        if not args.only_summaries:
            phase_parse_index()
            if args.skip_reprocess:
                _log("Phase 3/4 — skipped (--skip-reprocess)")
            else:
                phase_reprocess()
        else:
            _log("Phases 1-3 — skipped (--only-summaries)")
        phase_summaries(force=args.force_summaries)
    except KeyboardInterrupt:
        _log("Interrupted — progress so far is saved; re-run to continue.")
        sys.exit(130)
    _log(f"=== Done in {(time.time()-t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()
