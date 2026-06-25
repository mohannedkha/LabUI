#!/usr/bin/env python3
"""
Repair stale absolute PDF paths after moving or copying the project.

`papers.source_pdf` in the index DB (and the "source_pdf" field in
data/parsed/*.json) stores the ABSOLUTE path captured at index time. If the
project moves — e.g. ~/Apps/Rag/LabUI → ~/LabUI, or onto another machine — those
paths dangle and the PDF viewer 404s.

This rewrites each stored path to the file's CURRENT location, found by filename
under the configured papers dir (config.PAPERS_PDF_DIR). It's independent of
whatever the old path was, idempotent, and safe to re-run.

Run from the repo root with the project's venv:
    submission_strategy/.venv/bin/python Local_Rag/rag/fix_pdf_paths.py --dry-run
    submission_strategy/.venv/bin/python Local_Rag/rag/fix_pdf_paths.py
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from config import DB_PATH, PARSED_DIR, PAPERS_PDF_DIR


def _index_papers_dir() -> dict[str, str]:
    """Map filename -> absolute path for every PDF under the current papers dir."""
    idx: dict[str, str] = {}
    if PAPERS_PDF_DIR.exists():
        for p in PAPERS_PDF_DIR.rglob("*"):
            if p.is_file() and p.suffix.lower() == ".pdf" and not p.name.startswith("._"):
                idx.setdefault(p.name, str(p.resolve()))
    return idx


def _resolve(src: str | None, idx: dict[str, str]) -> str | None:
    """Current absolute path for a stored source_pdf, or None if not found."""
    if not src:
        return None
    p = Path(src)
    if p.exists():
        return str(p.resolve())          # already valid
    return idx.get(Path(src).name)       # relocate by filename


def fix_db(idx: dict[str, str], dry: bool) -> tuple[int, int, int]:
    if not DB_PATH.exists():
        print(f"  (no index db at {DB_PATH})")
        return (0, 0, 0)
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    rows = conn.execute("SELECT paper_id, source_pdf FROM papers").fetchall()
    fixed = ok = missing = 0
    for pid, src in rows:
        new = _resolve(src, idx)
        if new is None:
            missing += 1
            print(f"  ! no matching file for: {Path(src).name if src else pid}")
            continue
        if new == src:
            ok += 1
            continue
        if not dry:
            conn.execute("UPDATE papers SET source_pdf = ? WHERE paper_id = ?", (new, pid))
        fixed += 1
    if not dry:
        conn.commit()
    conn.close()
    return fixed, ok, missing


def fix_parsed(idx: dict[str, str], dry: bool) -> int:
    """Also update the cached parsed JSON so a future re-index stays consistent."""
    if not PARSED_DIR.exists():
        return 0
    fixed = 0
    for jf in PARSED_DIR.glob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        new = _resolve(data.get("source_pdf"), idx)
        if new and new != data.get("source_pdf"):
            data["source_pdf"] = new
            if not dry:
                jf.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            fixed += 1
    return fixed


def main() -> None:
    ap = argparse.ArgumentParser(description="Repair stale PDF paths after a move.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would change without writing anything")
    args = ap.parse_args()

    print(f"Papers dir : {PAPERS_PDF_DIR}")
    print(f"Index db   : {DB_PATH}")
    idx = _index_papers_dir()
    print(f"Found {len(idx)} PDF(s) in the current papers dir.\n")

    db_fixed, ok, missing = fix_db(idx, args.dry_run)
    parsed_fixed = fix_parsed(idx, args.dry_run)

    verb = "Would fix" if args.dry_run else "Fixed"
    print(f"\n{verb}: {db_fixed} db row(s), {parsed_fixed} parsed file(s). "
          f"{ok} already valid, {missing} unmatched.")
    if args.dry_run:
        print("(dry run — nothing written; re-run without --dry-run to apply.)")
    elif missing:
        print("Unmatched papers keep their old path — make sure their PDF is in the papers dir.")


if __name__ == "__main__":
    main()
