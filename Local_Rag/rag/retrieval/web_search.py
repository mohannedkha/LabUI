"""
DuckDuckGo web search for query augmentation.
Results are filtered to prioritise academic/scientific sources and
excluded from citation validation (they use [web:N] markers, not paper_ids).
"""
import logging
import re
from urllib.parse import urlparse

log = logging.getLogger(__name__)

# Domains that almost never produce useful scientific content
_BLOCKLIST = {
    "linkedin.com", "facebook.com", "twitter.com", "x.com",
    "instagram.com", "reddit.com", "quora.com", "pinterest.com",
    "youtube.com", "tiktok.com", "amazon.com", "ebay.com",
}

# Domains that score higher in ranking
_ACADEMIC_DOMAINS = {
    "arxiv.org", "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
    "scholar.google.com", "semanticscholar.org", "sciencedirect.com",
    "springer.com", "wiley.com", "nature.com", "science.org",
    "acs.org", "rsc.org", "mdpi.com", "frontiersin.org",
    "tandfonline.com", "pubs.acs.org", "iopscience.iop.org",
    "researchgate.net", "academia.edu", "ssrn.com",
}


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lstrip("www.")
    except Exception:
        return ""


def _score(result: dict) -> int:
    dom = _domain(result.get("href", ""))
    if dom in _ACADEMIC_DOMAINS:
        return 2
    if any(dom.endswith(a) for a in _ACADEMIC_DOMAINS):
        return 1
    return 0


def web_search(query: str, max_results: int = 6, academic_boost: bool = True) -> list[dict]:
    """
    Search DuckDuckGo and return up to max_results results.

    Each result dict has:
        index   : int  (1-based, used for [web:N] citations)
        title   : str
        url     : str
        snippet : str
    """
    try:
        from ddgs import DDGS
    except ImportError:
        log.warning("ddgs not installed — web search disabled")
        return []

    # Build query: for academic domains, append a filter hint
    search_q = query
    if academic_boost:
        search_q = (
            f"{query} "
            "site:arxiv.org OR site:pubmed.ncbi.nlm.nih.gov OR "
            "site:sciencedirect.com OR site:springer.com OR "
            "site:nature.com OR site:mdpi.com OR site:wiley.com OR "
            "site:researchgate.net OR site:semanticscholar.org"
        )

    raw = []
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(search_q, max_results=max_results * 2))
    except Exception as e:
        log.warning("DuckDuckGo search failed: %s", e)
        # Retry without academic filter
        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results * 2))
        except Exception as e2:
            log.error("DuckDuckGo retry failed: %s", e2)
            return []

    # Filter and rank
    filtered = [r for r in raw if _domain(r.get("href", "")) not in _BLOCKLIST]
    filtered.sort(key=_score, reverse=True)

    results = []
    for i, r in enumerate(filtered[:max_results], start=1):
        results.append({
            "index":   i,
            "title":   (r.get("title") or "").strip(),
            "url":     (r.get("href")  or "").strip(),
            "snippet": (r.get("body")  or "").strip(),
        })

    return results
