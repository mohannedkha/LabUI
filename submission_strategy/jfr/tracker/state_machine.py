"""Submission tracker: state machine and CRUD operations."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

from jfr.db.schema import VALID_TRANSITIONS, TERMINAL_STATES


class InvalidTransitionError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gen_id(prefix: str) -> str:
    short = uuid.uuid4().hex[:8]
    return f"{prefix}_{short}"


# ── Manuscript ─────────────────────────────────────────────────────────────────

def create_manuscript(
    conn: sqlite3.Connection,
    title: str,
    abstract: str,
    principal_claim: str,
    techniques: list[str] = None,
    figures: list[str] = None,
    abstract_format: str = "flat",
    bibtex_key: Optional[str] = None,
    ms_id: Optional[str] = None,
) -> str:
    ms_id = ms_id or _gen_id("ms")
    now = _now()
    conn.execute(
        """INSERT INTO manuscript
           (id, title, abstract, abstract_format, principal_claim,
            techniques_json, figures_json, bibtex_key, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            ms_id, title, abstract, abstract_format, principal_claim,
            json.dumps(techniques or []),
            json.dumps(figures or []),
            bibtex_key, now, now,
        ),
    )
    conn.commit()
    return ms_id


def get_manuscript(conn: sqlite3.Connection, ms_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM manuscript WHERE id=?", (ms_id,)).fetchone()
    if not row:
        return None
    return dict(row)


def list_manuscripts(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM manuscript ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


# ── Journal ────────────────────────────────────────────────────────────────────

def upsert_journal(conn: sqlite3.Connection, journal: dict) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO journal
           (id, name, publisher, publisher_family, issn_print, issn_electronic,
            is_fully_oa, is_hybrid_oa, impact_factor, society_affiliation,
            submission_url, abstract_format, scope_statement, metadata_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            journal["id"], journal["name"], journal["publisher"],
            journal.get("publisher_family", journal["publisher"].lower().split()[0]),
            journal.get("issn_print"), journal.get("issn_electronic"),
            int(journal.get("is_fully_oa", False)),
            int(journal.get("is_hybrid_oa", False)),
            journal.get("impact_factor"),
            journal.get("society_affiliation"),
            journal.get("submission_url"),
            journal.get("abstract_format"),
            journal.get("scope_statement"),
            json.dumps(journal.get("metadata", {})),
        ),
    )
    conn.commit()


def load_journals_from_yaml(conn: sqlite3.Connection, yaml_path) -> int:
    import yaml
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    count = 0
    for j in data.get("journals", []):
        upsert_journal(conn, j)
        count += 1
    return count


# ── Submission ─────────────────────────────────────────────────────────────────

def create_submission(
    conn: sqlite3.Connection,
    manuscript_id: str,
    journal_id: str,
    initial_state: str = "drafting",
    sub_id: Optional[str] = None,
) -> str:
    sub_id = sub_id or _gen_id("sub")
    now = _now()
    conn.execute(
        """INSERT INTO submission
           (id, manuscript_id, journal_id, current_state, created_at, updated_at)
           VALUES (?,?,?,?,?,?)""",
        (sub_id, manuscript_id, journal_id, initial_state, now, now),
    )
    conn.execute(
        """INSERT INTO submission_transition
           (submission_id, from_state, to_state, transitioned_at)
           VALUES (?,?,?,?)""",
        (sub_id, None, initial_state, now),
    )
    conn.commit()
    return sub_id


def transition_submission(
    conn: sqlite3.Connection,
    sub_id: str,
    to_state: str,
    notes: Optional[str] = None,
) -> None:
    row = conn.execute("SELECT current_state FROM submission WHERE id=?", (sub_id,)).fetchone()
    if not row:
        raise ValueError(f"Submission {sub_id!r} not found")
    from_state = row["current_state"]
    if from_state in TERMINAL_STATES:
        raise InvalidTransitionError(f"Cannot transition from terminal state {from_state!r}")
    allowed = VALID_TRANSITIONS.get(from_state, [])
    if to_state not in allowed:
        raise InvalidTransitionError(
            f"Invalid transition {from_state!r} → {to_state!r}. "
            f"Allowed: {allowed}"
        )
    now = _now()
    submitted_at_update = ""
    if to_state == "submitted":
        submitted_at_update = ", submitted_at=?"
    params_update = [to_state, now, sub_id]
    if to_state == "submitted":
        params_update = [to_state, now, now, sub_id]
    conn.execute(
        f"UPDATE submission SET current_state=?, updated_at=?{submitted_at_update} WHERE id=?",
        params_update,
    )
    conn.execute(
        """INSERT INTO submission_transition
           (submission_id, from_state, to_state, transitioned_at, notes)
           VALUES (?,?,?,?,?)""",
        (sub_id, from_state, to_state, now, notes),
    )
    conn.commit()


def get_submission(conn: sqlite3.Connection, sub_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM submission WHERE id=?", (sub_id,)).fetchone()
    if not row:
        return None
    sub = dict(row)
    sub["transitions"] = [
        dict(r) for r in conn.execute(
            "SELECT * FROM submission_transition WHERE submission_id=? ORDER BY transitioned_at",
            (sub_id,),
        ).fetchall()
    ]
    sub["reviewer_comments"] = [
        dict(r) for r in conn.execute(
            "SELECT * FROM reviewer_comment WHERE submission_id=? ORDER BY round, reviewer_number",
            (sub_id,),
        ).fetchall()
    ]
    return sub


def list_active_submissions(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """SELECT s.*, m.title as manuscript_title, j.name as journal_name
           FROM submission s
           JOIN manuscript m ON m.id = s.manuscript_id
           JOIN journal j ON j.id = s.journal_id
           WHERE s.current_state NOT IN ('accepted','rejected_desk','rejected_post_review','withdrawn')
           ORDER BY s.updated_at ASC""",
    ).fetchall()
    return [dict(r) for r in rows]


def list_submissions_for_manuscript(conn: sqlite3.Connection, manuscript_id: str) -> list[dict]:
    rows = conn.execute(
        """SELECT s.*, j.name as journal_name
           FROM submission s JOIN journal j ON j.id=s.journal_id
           WHERE s.manuscript_id=? ORDER BY s.created_at""",
        (manuscript_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def list_submissions_for_journal(conn: sqlite3.Connection, journal_id: str) -> list[dict]:
    rows = conn.execute(
        """SELECT s.*, m.title as manuscript_title
           FROM submission s JOIN manuscript m ON m.id=s.manuscript_id
           WHERE s.journal_id=? ORDER BY s.created_at""",
        (journal_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def update_reviewer_comment(
    conn: sqlite3.Connection,
    comment_id: int,
    review_round: Optional[int] = None,
    reviewer_number: Optional[int] = None,
    comment_text: Optional[str] = None,
    response_text: Optional[str] = None,
    received_at: Optional[str] = None,
) -> None:
    fields, params = [], []
    if review_round is not None:
        fields.append("round=?"); params.append(review_round)
    if reviewer_number is not None:
        fields.append("reviewer_number=?"); params.append(reviewer_number)
    if comment_text is not None:
        fields.append("comment_text=?"); params.append(comment_text)
    if response_text is not None:
        fields.append("response_text=?"); params.append(response_text)
    if received_at is not None:
        fields.append("received_at=?"); params.append(received_at)
    if not fields:
        return
    params.append(comment_id)
    conn.execute(f"UPDATE reviewer_comment SET {', '.join(fields)} WHERE id=?", params)
    conn.commit()


def update_manuscript(
    conn: sqlite3.Connection,
    ms_id: str,
    title: Optional[str] = None,
    abstract: Optional[str] = None,
    principal_claim: Optional[str] = None,
    techniques: Optional[list[str]] = None,
    bibtex_key: Optional[str] = None,
) -> None:
    import json as _json
    fields, params = [], []
    now = _now()
    if title is not None:
        fields.append("title=?"); params.append(title)
    if abstract is not None:
        fields.append("abstract=?"); params.append(abstract)
    if principal_claim is not None:
        fields.append("principal_claim=?"); params.append(principal_claim)
    if techniques is not None:
        fields.append("techniques_json=?"); params.append(_json.dumps(techniques))
    if bibtex_key is not None:
        fields.append("bibtex_key=?"); params.append(bibtex_key)
    fields.append("updated_at=?"); params.append(now)
    params.append(ms_id)
    conn.execute(f"UPDATE manuscript SET {', '.join(fields)} WHERE id=?", params)
    conn.commit()


def add_reviewer_comment(
    conn: sqlite3.Connection,
    sub_id: str,
    reviewer_number: int,
    comment_text: str,
    round: int = 1,
    received_at: Optional[str] = None,
) -> None:
    conn.execute(
        """INSERT INTO reviewer_comment
           (submission_id, round, reviewer_number, comment_text, received_at)
           VALUES (?,?,?,?,?)""",
        (sub_id, round, reviewer_number, comment_text, received_at or _now()),
    )
    conn.commit()
