#!/usr/bin/env python3
"""
Repair messy paper titles + fill in DOI/year/authors by matching each paper
against Crossref (the authoritative, free DOI registry).

Two phases, kept separate so matches can be reviewed before anything is written:

  python tools/refresh_metadata.py fetch        # query Crossref -> proposals.json (no DB writes)
  python tools/refresh_metadata.py apply         # apply AUTO tier only (backs up rag.db first)
  python tools/refresh_metadata.py apply --include-review   # also apply REVIEW tier

Tiers (by overlap coefficient of significant tokens + count of shared tokens):
  AUTO   ov >= 0.85 and shared >= 5   -> high precision, safe to auto-apply
  REVIEW ov >= 0.60 and shared >= 3   -> likely-but-uncertain, needs human eyes
  SKIP   everything else / no candidate (left untouched)
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

MAILTO = "you@example.com"
PROPOSALS = config.DB_PATH.parent / "metadata_proposals.json"

_STOP = set(
    "a an the of and or in on for to with by from as at into via using study studies "
    "based effect effects role new novel toward towards their its this that".split()
)


def _tokens(s: str) -> list[str]:
    return [t for t in re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split()
            if t not in _STOP and len(t) > 1]


def overlap(a: str, b: str) -> tuple[float, int]:
    A, B = set(_tokens(a)), set(_tokens(b))
    if not A or not B:
        return 0.0, 0
    inter = A & B
    return len(inter) / min(len(A), len(B)), len(inter)


def clean_title_for_query(t: str) -> str:
    t = t or ""
    t = re.sub(r"(?i)cite this:.*?\d{4},?\s*\d+,?\s*[\d–-]+", "", t)
    t = re.sub(r"(?i)pubs\.acs\.org\S*", "", t)
    t = re.sub(r"(?i)^(nano\w+|materials|molecules|polymers|membranes)\s+(article|review)\b", "", t)
    t = re.sub(r"(?i)advances in colloid and interface science\s*\d+\s*\(\d{4}\)\s*[\d–-]+", "", t)
    # filename-style slugs: dashes -> spaces
    if t.count("-") >= 3 and " " not in t.strip():
        t = t.replace("-", " ")
    t = re.sub(r"\s{2,}", " ", t).strip(" |·-")
    return t


def clean_crossref_title(t: str) -> str:
    t = re.sub(r"<[^>]+>", "", t or "")      # strip <sub>/<sup>/<i> tags
    t = html.unescape(t)
    return re.sub(r"\s{2,}", " ", t).strip()


def _is_real_work(it: dict) -> bool:
    if it.get("type") == "component":        # supplementary material record
        return False
    if re.search(r"\.s\d{3,}$", it.get("DOI") or ""):
        return False
    return True


def _authors(it: dict) -> str:
    out = []
    for a in it.get("author", []) or []:
        fam, giv = a.get("family"), a.get("given")
        if fam and giv:
            out.append(f"{fam}, {giv}")
        elif fam:
            out.append(fam)
    return "; ".join(out)


def _year(it: dict):
    for key in ("published-print", "published-online", "issued"):
        dp = (it.get(key, {}) or {}).get("date-parts", [[None]])
        if dp and dp[0] and dp[0][0]:
            return dp[0][0]
    return None


def classify(ov: float, n: int) -> str:
    if ov >= 0.85 and n >= 5:
        return "AUTO"
    if ov >= 0.60 and n >= 3:
        return "REVIEW"
    return "SKIP"


def fetch():
    con = sqlite3.connect(str(config.DB_PATH))
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT paper_id, title, source_pdf FROM papers ORDER BY title").fetchall()
    con.close()

    s = requests.Session()
    s.headers.update({"User-Agent": f"LabUI-RAG/1.0 (mailto:{MAILTO})"})

    proposals = []
    counts = {"AUTO": 0, "REVIEW": 0, "SKIP": 0}
    for i, r in enumerate(rows, 1):
        q = clean_title_for_query(r["title"])
        best = None
        try:
            resp = s.get(
                "https://api.crossref.org/works",
                params={"query.bibliographic": q[:300], "rows": 5,
                        "select": "DOI,title,author,issued,published-print,published-online,type"},
                timeout=25,
            )
            items = [it for it in resp.json().get("message", {}).get("items", []) if _is_real_work(it)]
        except Exception as e:
            items = []
            print(f"  [{i}] ERR {e}", file=sys.stderr)
        for it in items:
            ct = clean_crossref_title((it.get("title") or [""])[0])
            ov, n = overlap(q, ct)
            if best is None or ov > best["ov"]:
                best = {"ov": ov, "n": n, "doi": it.get("DOI"),
                        "new_title": ct, "year": _year(it), "authors": _authors(it)}
        tier = classify(best["ov"], best["n"]) if best else "SKIP"
        counts[tier] += 1
        proposals.append({
            "paper_id": r["paper_id"], "old_title": r["title"], "query": q, "tier": tier,
            **(best or {"ov": 0.0, "n": 0, "doi": None, "new_title": None,
                        "year": None, "authors": None}),
        })
        if i % 25 == 0:
            print(f"  …{i}/{len(rows)}  {counts}")
        time.sleep(0.4)

    PROPOSALS.write_text(json.dumps(proposals, indent=2, ensure_ascii=False))
    print(f"\nDONE  {counts}\nWrote {PROPOSALS}")


def apply(include_review: bool):
    proposals = json.loads(PROPOSALS.read_text())
    tiers = {"AUTO"} | ({"REVIEW"} if include_review else set())
    to_apply = [p for p in proposals if p["tier"] in tiers and p["new_title"]]

    backup = config.DB_PATH.with_suffix(f".db.pre_metadata_{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(config.DB_PATH, backup)
    print(f"Backed up rag.db -> {backup.name}")

    con = sqlite3.connect(str(config.DB_PATH))
    cols = {r[1] for r in con.execute("PRAGMA table_info(papers)")}
    if "doi" not in cols:
        con.execute("ALTER TABLE papers ADD COLUMN doi TEXT")
    if "title_original" not in cols:
        con.execute("ALTER TABLE papers ADD COLUMN title_original TEXT")
    con.commit()

    n = 0
    for p in to_apply:
        con.execute(
            "UPDATE papers SET "
            "title_original = COALESCE(title_original, title), "
            "title = ?, doi = ?, "
            "year = COALESCE(?, year), "
            "authors = CASE WHEN ? <> '' THEN ? ELSE authors END "
            "WHERE paper_id = ?",
            (p["new_title"], p["doi"], p["year"],
             p["authors"] or "", p["authors"] or "", p["paper_id"]),
        )
        n += 1
    con.commit()
    con.close()
    print(f"Applied {n} updates (tiers={sorted(tiers)}). Original titles saved in papers.title_original.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["fetch", "apply"])
    ap.add_argument("--include-review", action="store_true")
    a = ap.parse_args()
    if a.phase == "fetch":
        fetch()
    else:
        apply(a.include_review)
