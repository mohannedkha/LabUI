"""SQLite schema creation and migration."""

import sqlite3
from pathlib import Path

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS manuscript (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    abstract        TEXT NOT NULL,
    abstract_format TEXT CHECK (abstract_format IN ('flat', 'structured_hmf')),
    principal_claim TEXT NOT NULL,
    techniques_json TEXT NOT NULL DEFAULT '[]',
    figures_json    TEXT NOT NULL DEFAULT '[]',
    bibtex_key      TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS journal (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    publisher           TEXT NOT NULL,
    publisher_family    TEXT NOT NULL,
    issn_print          TEXT,
    issn_electronic     TEXT,
    is_fully_oa         INTEGER NOT NULL DEFAULT 0 CHECK (is_fully_oa IN (0,1)),
    is_hybrid_oa        INTEGER NOT NULL DEFAULT 0 CHECK (is_hybrid_oa IN (0,1)),
    impact_factor       REAL,
    society_affiliation TEXT,
    submission_url      TEXT,
    abstract_format     TEXT,
    scope_statement     TEXT,
    last_corpus_refresh TEXT,
    metadata_json       TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS corpus_article (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    journal_id      TEXT NOT NULL REFERENCES journal(id),
    doi             TEXT UNIQUE,
    title           TEXT,
    abstract        TEXT,
    keywords_json   TEXT NOT NULL DEFAULT '[]',
    topics_json     TEXT NOT NULL DEFAULT '[]',
    published_date  TEXT,
    vector_id       TEXT,
    embedding_model TEXT,
    ingested_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_corpus_journal ON corpus_article(journal_id);
CREATE INDEX IF NOT EXISTS idx_corpus_published ON corpus_article(published_date);

CREATE TABLE IF NOT EXISTS submission (
    id              TEXT PRIMARY KEY,
    manuscript_id   TEXT NOT NULL REFERENCES manuscript(id),
    journal_id      TEXT NOT NULL REFERENCES journal(id),
    current_state   TEXT NOT NULL,
    submitted_at    TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sub_manuscript ON submission(manuscript_id);
CREATE INDEX IF NOT EXISTS idx_sub_journal    ON submission(journal_id);
CREATE INDEX IF NOT EXISTS idx_sub_state      ON submission(current_state);

CREATE TABLE IF NOT EXISTS submission_transition (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id   TEXT NOT NULL REFERENCES submission(id),
    from_state      TEXT,
    to_state        TEXT NOT NULL,
    transitioned_at TEXT NOT NULL,
    notes           TEXT
);
CREATE INDEX IF NOT EXISTS idx_trans_sub ON submission_transition(submission_id);

CREATE TABLE IF NOT EXISTS reviewer_comment (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id   TEXT NOT NULL REFERENCES submission(id),
    round           INTEGER NOT NULL DEFAULT 1,
    reviewer_number INTEGER NOT NULL,
    comment_text    TEXT NOT NULL,
    response_text   TEXT,
    received_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recommendation_event (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    manuscript_id   TEXT NOT NULL REFERENCES manuscript(id),
    recommended_at  TEXT NOT NULL,
    results_json    TEXT NOT NULL,
    chosen_journal  TEXT REFERENCES journal(id)
);

CREATE TABLE IF NOT EXISTS model_version (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    installed_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiment (
    id              TEXT PRIMARY KEY,
    manuscript_id   TEXT REFERENCES manuscript(id) ON DELETE SET NULL,
    name            TEXT NOT NULL,
    ran_on          TEXT,
    scheduled_for   TEXT,
    status          TEXT NOT NULL DEFAULT 'planned',
    objective       TEXT,
    methodology     TEXT,
    conditions_json TEXT NOT NULL DEFAULT '{}',
    equipment       TEXT,
    observations    TEXT,
    results_md      TEXT,
    notes_md        TEXT,
    tags_json       TEXT NOT NULL DEFAULT '[]',
    linked_papers_json TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_experiment_manuscript    ON experiment(manuscript_id);
CREATE INDEX IF NOT EXISTS idx_experiment_status        ON experiment(status);
CREATE INDEX IF NOT EXISTS idx_experiment_ran_on        ON experiment(ran_on);
CREATE INDEX IF NOT EXISTS idx_experiment_scheduled_for ON experiment(scheduled_for);
"""

EXPERIMENT_STATUSES = ["planned", "in_progress", "done", "failed", "abandoned"]

VALID_TRANSITIONS: dict[str, list[str]] = {
    "drafting":                   ["internal_review", "withdrawn"],
    "internal_review":            ["awaiting_supervisor_approval", "drafting", "withdrawn"],
    "awaiting_supervisor_approval": ["submitted", "internal_review", "withdrawn"],
    "submitted":                  ["under_review", "rejected_desk", "withdrawn"],
    "under_review":               ["revision_requested_minor", "revision_requested_major",
                                   "accepted", "rejected_post_review", "withdrawn"],
    "revision_requested_minor":   ["revising", "withdrawn"],
    "revision_requested_major":   ["revising", "withdrawn"],
    "revising":                   ["resubmitted", "withdrawn"],
    "resubmitted":                ["under_review", "accepted", "rejected_post_review", "withdrawn"],
    "accepted":                   [],
    "rejected_desk":              [],
    "rejected_post_review":       [],
    "withdrawn":                  [],
}

TERMINAL_STATES = {"accepted", "rejected_desk", "rejected_post_review", "withdrawn"}


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(DDL)
    conn.commit()
    return conn


def get_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn
