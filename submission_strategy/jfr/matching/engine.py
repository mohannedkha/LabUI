"""Matching engine: hybrid dense + lexical scoring with policy filter."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from jfr.config import Policy


@dataclass
class ManuscriptInput:
    title: str
    abstract: str
    principal_claim: str
    techniques: list[str] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)
    abstract_format: str = "flat"

    def query_text(self) -> str:
        return f"{self.title}. {self.abstract}"

    def lexical_tokens(self) -> list[str]:
        combined = " ".join([self.abstract, self.principal_claim] + self.techniques)
        return combined.lower().split()


@dataclass
class ScoreDecomposition:
    dense_abstract: float = 0.0
    lexical_techniques: float = 0.0
    principal_claim: float = 0.0
    topk_dispersion_penalty: float = 0.0


@dataclass
class PolicyResult:
    passed: bool
    reason: str = ""


@dataclass
class RecommendationEntry:
    rank: int
    journal_id: str
    journal_name: str
    score: float
    decomposition: ScoreDecomposition
    policy: PolicyResult
    nearest_articles: list[dict] = field(default_factory=list)
    user_history: dict = field(default_factory=dict)
    median_turnaround_days: Optional[int] = None
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "journal_id": self.journal_id,
            "journal_name": self.journal_name,
            "score": self.score,
            "decomposition": {
                "dense_abstract": self.decomposition.dense_abstract,
                "lexical_techniques": self.decomposition.lexical_techniques,
                "principal_claim": self.decomposition.principal_claim,
                "topk_dispersion_penalty": self.decomposition.topk_dispersion_penalty,
            },
            "policy_passed": self.policy.passed,
            "policy_reason": self.policy.reason,
            "nearest_articles": self.nearest_articles,
            "user_history": self.user_history,
            "median_turnaround_days": self.median_turnaround_days,
            "rationale": self.rationale,
        }


def _build_bm25_corpus(conn: sqlite3.Connection, journal_id: str, limit: int = 500) -> tuple[BM25Okapi, list[dict]]:
    rows = conn.execute(
        """SELECT id, doi, title, abstract, keywords_json, topics_json
           FROM corpus_article
           WHERE journal_id=? AND abstract IS NOT NULL AND abstract != ''
           ORDER BY published_date DESC LIMIT ?""",
        (journal_id, limit),
    ).fetchall()
    docs = []
    corpus = []
    for r in rows:
        tokens = (
            (r["title"] or "") + " " +
            (r["abstract"] or "") + " " +
            " ".join(json.loads(r["keywords_json"])) + " " +
            " ".join(json.loads(r["topics_json"]))
        ).lower().split()
        corpus.append(tokens)
        docs.append({"doi": r["doi"], "title": r["title"], "id": r["id"]})
    if not corpus:
        return None, []
    return BM25Okapi(corpus), docs


def score_journal_dense(
    qdrant_client,
    collection_name: str,
    query_vector: np.ndarray,   # specter2 (768-dim) — abstract query
    claim_vector: np.ndarray,   # specter2 (768-dim) — claim query (same space as corpus)
    top_k: int = 25,
) -> tuple[float, float, float, list[dict]]:
    """
    Both vectors must be in the corpus embedding space (specter2, 768-dim).
    Returns (dense_abstract_score, principal_claim_score, dispersion_penalty, nearest_articles).
    """
    result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        limit=top_k,
        with_payload=True,
    )
    hits = result.points
    if not hits:
        return 0.0, 0.0, 0.0, []

    scores = [h.score for h in hits]
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores)) if len(scores) > 1 else 0.0
    dispersion_penalty = -std_score * 0.5  # penalise narrow cluster — narrow match = tangential fit

    claim_result = qdrant_client.query_points(
        collection_name=collection_name,
        query=claim_vector.tolist(),
        limit=top_k,
        with_payload=True,
    )
    claim_hits = claim_result.points
    claim_scores = [h.score for h in claim_hits] if claim_hits else [0.0]
    claim_mean = float(np.mean(claim_scores))

    nearest = [
        {"doi": h.payload.get("doi"), "similarity": round(h.score, 4)}
        for h in hits[:5]
    ]
    return mean_score, claim_mean, dispersion_penalty, nearest


def score_journal_lexical(
    bm25: BM25Okapi,
    query_tokens: list[str],
) -> float:
    if bm25 is None or not query_tokens:
        return 0.0
    raw_scores = bm25.get_scores(query_tokens)
    max_score = float(raw_scores.max()) if raw_scores.size else 0.0
    if max_score == 0:
        return 0.0
    return float(np.mean(np.sort(raw_scores)[-25:])) / max_score


def _median_turnaround(conn: sqlite3.Connection, journal_id: str) -> Optional[int]:
    """Median submission→first-decision days from our own tracker history for this journal."""
    from datetime import datetime
    rows = conn.execute(
        """SELECT t.transitioned_at, s.submitted_at
           FROM submission s
           JOIN submission_transition t ON t.submission_id = s.id
           WHERE s.journal_id = ?
             AND s.submitted_at IS NOT NULL
             AND t.to_state IN ('accepted', 'rejected_desk', 'rejected_post_review')""",
        (journal_id,),
    ).fetchall()
    days_list = []
    for r in rows:
        try:
            sub = datetime.fromisoformat(r["submitted_at"].replace("Z", "+00:00"))
            dec = datetime.fromisoformat(r["transitioned_at"].replace("Z", "+00:00"))
            d = (dec - sub).days
            if d >= 0:
                days_list.append(d)
        except Exception:
            pass
    if not days_list:
        return None
    days_list.sort()
    return days_list[len(days_list) // 2]


def build_user_history(conn: sqlite3.Connection, manuscript_id: str, journal_id: str) -> dict:
    rows = conn.execute(
        """SELECT s.current_state, s.submitted_at
           FROM submission s
           WHERE s.journal_id=? AND s.manuscript_id != ?
           ORDER BY s.created_at DESC""",
        (journal_id, manuscript_id),
    ).fetchall()
    accepted = sum(1 for r in rows if r["current_state"] == "accepted")
    rejected = sum(1 for r in rows if "rejected" in r["current_state"])
    return {
        "prior_submissions": len(rows),
        "prior_acceptances": accepted,
        "prior_rejections": rejected,
    }


def build_rationale(entry: RecommendationEntry) -> str:
    parts = []
    if entry.decomposition.dense_abstract >= 0.7:
        parts.append(f"strong abstract match (score {entry.decomposition.dense_abstract:.2f})")
    if entry.decomposition.lexical_techniques >= 0.5:
        parts.append("technique terminology overlap")
    hist = entry.user_history
    if hist.get("prior_acceptances", 0) > 0:
        parts.append(f"prior acceptance at this venue")
    if hist.get("prior_rejections", 0) > 0:
        parts.append(f"prior rejection — review rejection notes")
    if not parts:
        parts.append("moderate topical overlap across recent articles")
    return "; ".join(parts).capitalize() + "."


def recommend(
    manuscript: ManuscriptInput,
    conn: sqlite3.Connection,
    qdrant_client,
    policy: Policy,
    abstract_model: str,
    claim_model: str,
    top_n: int = 10,
    top_k: int = 25,
    manuscript_id: str = "",
) -> list[RecommendationEntry]:
    """
    Full recommendation pipeline. Returns ranked list of RecommendationEntry.
    """
    from jfr.corpus.embedder import embed_texts

    journals = conn.execute("SELECT * FROM journal").fetchall()

    # Both vectors use abstract_model (specter2, 768-dim) so they're in the same
    # space as the Qdrant corpus.  claim_model (bge-large, 1024-dim) is reserved
    # for a future non-Qdrant re-ranking pass.
    query_vec = embed_texts([manuscript.query_text()], abstract_model)[0]
    claim_vec = embed_texts([manuscript.principal_claim], abstract_model)[0]
    lex_tokens = manuscript.lexical_tokens()

    candidates: list[RecommendationEntry] = []

    for j in journals:
        jid = j["id"]

        # Policy filter (hard)
        excluded, excl_reason = policy.is_journal_excluded(jid)
        if excluded:
            continue

        oa_ok, oa_reason = policy.check_open_access(bool(j["is_fully_oa"]), bool(j["is_hybrid_oa"]))
        if not oa_ok:
            candidates.append(_policy_fail(j, oa_reason, len(candidates) + 1))
            continue

        if_ok, if_reason = policy.check_impact_factor(j["impact_factor"], journal_id=jid)
        if not if_ok:
            candidates.append(_policy_fail(j, if_reason, len(candidates) + 1))
            continue

        pub_ok, pub_reason = policy.check_publisher(j["publisher_family"])
        if not pub_ok:
            candidates.append(_policy_fail(j, pub_reason, len(candidates) + 1))
            continue

        collection = f"journal_{jid}"
        # Check collection exists
        try:
            info = qdrant_client.get_collection(collection)
            has_vectors = bool(info.points_count and info.points_count > 0)
        except Exception:
            has_vectors = False

        dense_abs, dense_claim, dispersion, nearest = (0.0, 0.0, 0.0, [])
        if has_vectors:
            dense_abs, dense_claim, dispersion, nearest = score_journal_dense(
                qdrant_client, collection, query_vec, claim_vec, top_k
            )
            # Enrich nearest articles with title from SQLite
            if nearest:
                dois = [a["doi"] for a in nearest if a.get("doi")]
                if dois:
                    placeholders = ",".join("?" * len(dois))
                    title_map = {
                        r["doi"]: r["title"]
                        for r in conn.execute(
                            f"SELECT doi, title FROM corpus_article WHERE doi IN ({placeholders})",
                            dois,
                        ).fetchall()
                    }
                    for a in nearest:
                        a["title"] = title_map.get(a.get("doi"), "")

        bm25, _ = _build_bm25_corpus(conn, jid)
        lex_score = score_journal_lexical(bm25, lex_tokens)

        decomp = ScoreDecomposition(
            dense_abstract=round(dense_abs, 4),
            lexical_techniques=round(lex_score, 4),
            principal_claim=round(dense_claim, 4),
            topk_dispersion_penalty=round(dispersion, 4),
        )

        history = build_user_history(conn, manuscript_id, jid)
        turnaround = _median_turnaround(conn, jid)

        entry = RecommendationEntry(
            rank=0,
            journal_id=jid,
            journal_name=j["name"],
            score=0.0,
            decomposition=decomp,
            policy=PolicyResult(passed=True),
            nearest_articles=nearest,
            user_history=history,
            median_turnaround_days=turnaround,
        )
        entry.score = round(
            (dense_abs * 0.5 + lex_score * 0.25 + dense_claim * 0.25) + dispersion, 4
        )
        entry.rationale = build_rationale(entry)
        candidates.append(entry)

    passed = [c for c in candidates if c.policy.passed]
    failed = [c for c in candidates if not c.policy.passed]
    passed.sort(key=lambda x: -x.score)

    results = []
    for i, e in enumerate(passed[:top_n]):
        e.rank = i + 1
        results.append(e)
    for e in failed:
        e.rank = len(results) + 1
        results.append(e)

    return results


def _policy_fail(journal_row, reason: str, rank: int) -> RecommendationEntry:
    return RecommendationEntry(
        rank=rank,
        journal_id=journal_row["id"],
        journal_name=journal_row["name"],
        score=0.0,
        decomposition=ScoreDecomposition(),
        policy=PolicyResult(passed=False, reason=reason),
    )
