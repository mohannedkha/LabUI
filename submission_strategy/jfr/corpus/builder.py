"""Corpus builder: fetches abstracts from CrossRef, OpenAlex, and RSS/Atom feeds."""

import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

CROSSREF_API = "https://api.crossref.org/works"
OPENALEX_API = "https://api.openalex.org/works"
UA = "jfr/0.1 (mailto:contact@example.com)"


def _cutoff_date(months: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=months * 30)
    return dt.strftime("%Y-%m-%d")


def fetch_crossref_articles(
    issn: str,
    months: int = 36,
    limit: int = 1000,
    progress_cb=None,
) -> list[dict]:
    """Retrieve up to `limit` recent articles for a journal ISSN from CrossRef."""
    cutoff = _cutoff_date(months)
    articles = []
    cursor = "*"
    per_page = min(100, limit)

    with httpx.Client(timeout=30, headers={"User-Agent": UA}) as client:
        while len(articles) < limit:
            params = {
                "filter": f"issn:{issn},from-pub-date:{cutoff}",
                "select": "DOI,title,abstract,published,author,subject",
                "rows": per_page,
                "cursor": cursor,
            }
            resp = client.get(CROSSREF_API, params=params)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("message", {}).get("items", [])
            if not items:
                break
            for item in items:
                pub_date = _extract_date(item.get("published"))
                if pub_date and pub_date < cutoff:
                    continue
                articles.append({
                    "doi": item.get("DOI"),
                    "title": _first(item.get("title")),
                    "abstract": _clean_abstract(item.get("abstract", "")),
                    "published_date": pub_date,
                    "keywords": item.get("subject", []),
                    "topics": [],
                })
            cursor = data.get("message", {}).get("next-cursor", "")
            if not cursor:
                break
            if progress_cb:
                progress_cb(len(articles))

    return articles[:limit]


def fetch_openalex_articles(
    issn: str,
    months: int = 36,
    limit: int = 1000,
    progress_cb=None,
) -> list[dict]:
    """Retrieve recent articles for a journal ISSN from OpenAlex."""
    cutoff = _cutoff_date(months)
    articles = []
    cursor = "*"
    per_page = min(200, limit)

    with httpx.Client(timeout=30, headers={"User-Agent": UA}) as client:
        while len(articles) < limit:
            params = {
                "filter": f"locations.source.issn:{issn},from_publication_date:{cutoff}",
                "select": "doi,title,abstract_inverted_index,publication_date,keywords,concepts",
                "per-page": per_page,
                "cursor": cursor,
            }
            resp = client.get(OPENALEX_API, params=params)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break
            for r in results:
                abstract = _reconstruct_abstract(r.get("abstract_inverted_index"))
                if not abstract:
                    continue
                articles.append({
                    "doi": _normalize_doi(r.get("doi")),
                    "title": r.get("title", ""),
                    "abstract": abstract,
                    "published_date": r.get("publication_date"),
                    "keywords": [k.get("keyword", "") for k in (r.get("keywords") or [])],
                    "topics": [c.get("display_name", "") for c in (r.get("concepts") or [])[:5]],
                })
            meta = data.get("meta", {})
            cursor = meta.get("next_cursor", "")
            if not cursor:
                break
            if progress_cb:
                progress_cb(len(articles))

    return articles[:limit]


def ingest_articles(
    conn: sqlite3.Connection,
    journal_id: str,
    articles: list[dict],
    embedding_model: Optional[str] = None,
) -> int:
    """Insert articles into corpus_article, skip existing DOIs. Returns count inserted."""
    now = datetime.now(timezone.utc).isoformat()
    inserted = 0
    for a in articles:
        try:
            conn.execute(
                """INSERT OR IGNORE INTO corpus_article
                   (journal_id, doi, title, abstract, keywords_json, topics_json,
                    published_date, embedding_model, ingested_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    journal_id,
                    a.get("doi"),
                    a.get("title", ""),
                    a.get("abstract", ""),
                    json.dumps(a.get("keywords", [])),
                    json.dumps(a.get("topics", [])),
                    a.get("published_date"),
                    embedding_model,
                    now,
                ),
            )
            if conn.execute("SELECT changes()").fetchone()[0]:
                inserted += 1
        except sqlite3.IntegrityError:
            pass
    conn.execute(
        "UPDATE journal SET last_corpus_refresh=? WHERE id=?",
        (now, journal_id),
    )
    conn.commit()
    return inserted


def fetch_rss_articles(feed_url: str, progress_cb=None) -> list[dict]:
    """Fetch recent articles from an RSS/Atom feed URL using feedparser.

    Returns articles in the same dict format as fetch_crossref_articles.
    RSS feeds typically return only the most recent 30-100 papers; use for
    incremental nightly refreshes, not initial corpus builds.
    """
    import feedparser
    import time as _time

    feed = feedparser.parse(feed_url)
    articles = []
    for entry in feed.entries:
        # Extract DOI
        doi = None
        for attr in ("prism_doi", "dc_identifier"):
            val = getattr(entry, attr, None)
            if val:
                doi = val.replace("doi:", "").strip()
                break
        if not doi:
            link = getattr(entry, "link", "") or ""
            if "doi.org/" in link:
                doi = link.split("doi.org/")[-1].split("?")[0].strip()

        # Parse publication date
        pub_date = None
        parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
        if parsed:
            try:
                pub_date = _time.strftime("%Y-%m-%d", parsed)
            except Exception:
                pass

        # Extract abstract text, stripping any HTML
        abstract = ""
        summary = getattr(entry, "summary", "") or ""
        if summary:
            abstract = re.sub(r"<[^>]+>", "", summary).strip()
        elif hasattr(entry, "content") and entry.content:
            abstract = re.sub(r"<[^>]+>", "", entry.content[0].get("value", "")).strip()

        title = re.sub(r"<[^>]+>", "", getattr(entry, "title", "") or "").strip()
        if not title and not abstract:
            continue

        articles.append({
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "published_date": pub_date,
            "keywords": [],
            "topics": [],
        })

    return articles


def corpus_stats(conn: sqlite3.Connection, journal_id: str) -> dict:
    """Return statistics for a journal's corpus."""
    row = conn.execute(
        """SELECT
             COUNT(*) as total,
             MIN(published_date) as oldest,
             MAX(published_date) as newest,
             SUM(CASE WHEN abstract != '' AND abstract IS NOT NULL THEN 1 ELSE 0 END) as with_abstract
           FROM corpus_article WHERE journal_id=?""",
        (journal_id,),
    ).fetchone()
    topics_rows = conn.execute(
        """SELECT topics_json FROM corpus_article
           WHERE journal_id=? AND topics_json != '[]' LIMIT 500""",
        (journal_id,),
    ).fetchall()
    topic_counts: dict[str, int] = {}
    for r in topics_rows:
        for t in json.loads(r["topics_json"]):
            topic_counts[t] = topic_counts.get(t, 0) + 1
    top_topics = sorted(topic_counts.items(), key=lambda x: -x[1])[:10]
    return {
        "total": row["total"],
        "with_abstract": row["with_abstract"],
        "oldest": row["oldest"],
        "newest": row["newest"],
        "top_topics": top_topics,
    }


# ── helpers ────────────────────────────────────────────────────────────────────


def _first(lst: Optional[list]) -> str:
    if lst:
        return lst[0]
    return ""


def _extract_date(pub: Optional[dict]) -> Optional[str]:
    if not pub:
        return None
    parts = pub.get("date-parts", [[]])[0]
    if len(parts) >= 3:
        return f"{parts[0]:04d}-{parts[1]:02d}-{parts[2]:02d}"
    if len(parts) == 2:
        return f"{parts[0]:04d}-{parts[1]:02d}-01"
    if len(parts) == 1:
        return f"{parts[0]:04d}-01-01"
    return None


def _clean_abstract(text: str) -> str:
    """Strip JATS XML tags from CrossRef abstracts."""
    import re
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _reconstruct_abstract(inv_idx: Optional[dict]) -> str:
    """Rebuild abstract from OpenAlex inverted index."""
    if not inv_idx:
        return ""
    pos_word: dict[int, str] = {}
    for word, positions in inv_idx.items():
        for pos in positions:
            pos_word[pos] = word
    return " ".join(pos_word[i] for i in sorted(pos_word))


def _normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    return doi.replace("https://doi.org/", "").lower()
