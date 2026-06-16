#!/usr/bin/env python3
"""
Second-pass metadata match for papers still missing a DOI after pass 1.

Improvements over pass 1:
  • uses the first chunk's section_name (often the real, clean title) as a query,
  • strips journal-header / preprint prefixes that polluted pass-1 queries,
  • validates the candidate's year against a year hint from the filename/title,
    which kills the year-mismatch false positives pass 1 flagged for review.

  python tools/refresh_metadata2.py fetch    # -> metadata_proposals2.json (no writes)
  python tools/refresh_metadata2.py apply     # apply AUTO2 only (backs up rag.db)
"""
from __future__ import annotations

import argparse
import html
import json
import os
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
PROPOSALS = config.DB_PATH.parent / "metadata_proposals2.json"

_STOP = set(
    "a an the of and or in on for to with by from as at into via using study studies "
    "based effect effects role new novel toward towards their its this that".split()
)


def _tokens(s):
    return [t for t in re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split()
            if t not in _STOP and len(t) > 1]


def overlap(a, b):
    A, B = set(_tokens(a)), set(_tokens(b))
    if not A or not B:
        return 0.0, 0
    inter = A & B
    return len(inter) / min(len(A), len(B)), len(inter)


def year_hint(*xs):
    for x in xs:
        m = re.findall(r"\b(19[5-9]\d|20[0-3]\d)\b", x or "")
        if m:
            return int(m[0])
    return None


def strip_prefix(t):
    t = t or ""
    # "Journal Name 104 (2003) 75-91 <title>"  ->  "<title>"
    t = re.sub(r"^.{0,80}?\(\s*(19|20)\d{2}\s*\)\s*[\divxlcdm]*\s*[–\-]\s*\d+\s*", "", t, flags=re.I)
    t = re.sub(r"(?i)^(week ending.*?\d{4}|journal pre-?proof|hosted by|title:)\s*", "", t)
    t = re.sub(r"(?i)arxiv:\s*\S+(\s*\[[^\]]+\])?(\s*\d+\s+\w+\s+\d{4})?", "", t)
    t = re.sub(r"(?i)cite this:.*?\d{4},?\s*\d+,?\s*[\d–-]+", "", t)
    t = re.sub(r"(?i)pubs\.acs\.org\S*", "", t)
    t = re.sub(r"(?i)^(nano\w+|materials|molecules|polymers|membranes|engineering)\s+(article|review)\b", "", t)
    t = re.sub(r"^[|│\s·\-–]+", "", t)
    if t.count("-") >= 3 and " " not in t.strip():
        t = t.replace("-", " ")
    return re.sub(r"\s{2,}", " ", t).strip(" |·-")


def clean_cr(t):
    return re.sub(r"\s{2,}", " ", html.unescape(re.sub(r"<[^>]+>", "", t or ""))).strip()


def _is_real(it):
    if it.get("type") in ("component", "peer-review"):
        return False
    return not re.search(r"\.s\d{3,}$|/review\d*$|/v\d+/review", it.get("DOI") or "")


def _authors(it):
    out = []
    for a in it.get("author", []) or []:
        fam, giv = a.get("family"), a.get("given")
        out.append(f"{fam}, {giv}" if fam and giv else (fam or ""))
    return "; ".join(x for x in out if x)


def _year(it):
    for k in ("published-print", "published-online", "issued"):
        dp = (it.get(k, {}) or {}).get("date-parts", [[None]])
        if dp and dp[0] and dp[0][0]:
            return dp[0][0]
    return None


def fetch():
    con = sqlite3.connect(str(config.DB_PATH)); con.row_factory = sqlite3.Row
    rows = con.execute("SELECT paper_id,title,source_pdf,authors FROM papers WHERE doi IS NULL ORDER BY title").fetchall()
    s = requests.Session(); s.headers.update({"User-Agent": f"LabUI-RAG/1.0 (mailto:{MAILTO})"})

    proposals, counts = [], {"AUTO2": 0, "REVIEW2": 0, "SKIP2": 0}
    for i, r in enumerate(rows, 1):
        sect = con.execute(
            "SELECT section_name FROM chunks WHERE paper_id=? ORDER BY position LIMIT 1",
            (r["paper_id"],)).fetchone()
        sect_title = sect["section_name"] if sect else ""
        title_q = strip_prefix(r["title"])
        # Prefer the section-name title when it looks like a real multiword title.
        queries = []
        if sect_title and len(_tokens(sect_title)) >= 4:
            queries.append(sect_title)
        if title_q:
            queries.append(title_q)
        if not queries:
            queries = [r["title"] or ""]
        truth = (sect_title or "") + " " + (title_q or "")
        yh = year_hint(os.path.basename(r["source_pdf"] or ""), r["title"], sect_title)

        cands = {}
        for q in queries[:2]:
            try:
                resp = s.get("https://api.crossref.org/works",
                             params={"query.bibliographic": q[:300], "rows": 5,
                                     "select": "DOI,title,author,issued,published-print,published-online,type"},
                             timeout=25)
                for it in resp.json().get("message", {}).get("items", []):
                    if _is_real(it) and it.get("DOI") not in cands:
                        cands[it.get("DOI")] = it
            except Exception as e:
                print(f"  [{i}] ERR {e}", file=sys.stderr)
            time.sleep(0.35)

        best = None
        for it in cands.values():
            ct = clean_cr((it.get("title") or [""])[0])
            ov, n = overlap(truth, ct)
            yr = _year(it)
            year_ok = (yh is None) or (yr is None) or abs(yr - yh) <= 1
            # year mismatch is a strong negative signal
            score = ov - (0.0 if year_ok else 0.5)
            if best is None or score > best["score"]:
                best = {"score": score, "ov": ov, "n": n, "year_ok": year_ok,
                        "doi": it.get("DOI"), "new_title": ct, "year": yr, "authors": _authors(it)}

        if not best:
            tier = "SKIP2"
        elif best["ov"] >= 0.85 and best["n"] >= 5 and best["year_ok"]:
            tier = "AUTO2"
        elif best["ov"] >= 0.9 and best["n"] >= 4 and (yh is not None and best["year_ok"]):
            tier = "AUTO2"          # fewer tokens OK when the year confirms it
        elif best["ov"] >= 0.6 and best["n"] >= 3:
            tier = "REVIEW2"
        else:
            tier = "SKIP2"
        counts[tier] += 1
        proposals.append({"paper_id": r["paper_id"], "old_title": r["title"], "tier": tier,
                          "year_hint": yh, **(best or {})})
        if i % 25 == 0:
            print(f"  …{i}/{len(rows)} {counts}")
    con.close()
    PROPOSALS.write_text(json.dumps(proposals, indent=2, ensure_ascii=False))
    print(f"\nDONE {counts}\nWrote {PROPOSALS}")


def apply():
    proposals = json.loads(PROPOSALS.read_text())
    todo = [p for p in proposals if p["tier"] == "AUTO2" and p.get("new_title")]
    backup = config.DB_PATH.with_suffix(f".db.pre_metadata2_{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(config.DB_PATH, backup)
    print(f"Backed up -> {backup.name}")
    con = sqlite3.connect(str(config.DB_PATH))
    n = 0
    for p in todo:
        con.execute(
            "UPDATE papers SET title_original=COALESCE(title_original,title), title=?, doi=?, "
            "year=COALESCE(?,year), authors=CASE WHEN ?<>'' THEN ? ELSE authors END WHERE paper_id=?",
            (p["new_title"], p["doi"], p["year"], p["authors"] or "", p["authors"] or "", p["paper_id"]))
        n += 1
    con.commit(); con.close()
    print(f"Applied {n} AUTO2 updates.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["fetch", "apply"])
    a = ap.parse_args()
    fetch() if a.phase == "fetch" else apply()
