"""Shared paper-summarisation core.

One implementation used by both the web layer (jfr.web.rag_routes) and the
bulk CLI (scripts/build_library.py): gather a paper's most informative text,
ask the local LLM for a clean prose summary, and cache it on papers.summary.

Reasoning models (Qwen3 / DeepSeek-R1 / QwQ) emit a <think>…</think> block; we
ask Ollama to skip it and strip any leaked block defensively.
"""
from __future__ import annotations

import re
import sqlite3

import requests

from config import OLLAMA_BASE_URL, GEN_MODEL, GEN_MODEL_PREFER

# Sections richest in summarisable content, tried first within the char budget.
_SUMMARY_PRIORITY = (
    "abstract", "summary", "introduction", "conclusion", "conclusions",
    "discussion", "results",
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# ── Reasoning-model handling ─────────────────────────────────────────────────
def is_reasoning_model(name: str) -> bool:
    n = (name or "").lower()
    return any(k in n for k in ("qwen3", "deepseek-r1", "-r1", "qwq", "reasoning", "thinking"))


def apply_thinking_off(payload: dict, model: str) -> dict:
    """Disable chain-of-thought for reasoning models (Ollama top-level `think`)."""
    if is_reasoning_model(model):
        payload["think"] = False
    return payload


def strip_think(text: str) -> str:
    """Remove <think>…</think> blocks from a completed response."""
    out = _THINK_BLOCK_RE.sub("", text or "")
    if "</think>" in out and "<think>" not in out:
        out = out.split("</think>", 1)[1]
    return out.strip()


# ── Model resolution (for standalone callers; the web layer has its own) ──────
def list_ollama_models(base_url: str = OLLAMA_BASE_URL) -> list[str]:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
        return [m.get("name", "") for m in (r.json().get("models") or []) if m.get("name")]
    except Exception:
        return []


def resolve_gen_model(base_url: str = OLLAMA_BASE_URL) -> str:
    """Pick the generator the same way the web app does: a pinned GEN_MODEL if
    present, else the first installed model matching GEN_MODEL_PREFER, else the
    first installed model. Returns "" when Ollama has nothing."""
    names = list_ollama_models(base_url)
    if not names:
        return ""
    if GEN_MODEL:
        if GEN_MODEL in names:
            return GEN_MODEL
        base = GEN_MODEL.split(":")[0]
        match = next((n for n in names if n.split(":")[0] == base), "")
        if match:
            return match
    for pref in GEN_MODEL_PREFER:
        match = next((n for n in names if pref.lower() in n.lower()), "")
        if match:
            return match
    return names[0]


# ── Summarisation ────────────────────────────────────────────────────────────
def ensure_summary_col(conn: sqlite3.Connection) -> None:
    have = {r[1] for r in conn.execute("PRAGMA table_info(papers)").fetchall()}
    if "summary" not in have:
        conn.execute("ALTER TABLE papers ADD COLUMN summary TEXT")
        conn.commit()


def gather_paper_text(conn: sqlite3.Connection, paper_id: str, max_chars: int = 8000) -> str:
    """Assemble representative paper text, leading with abstract/intro/conclusion
    and filling with the rest in reading order."""
    rows = conn.execute(
        "SELECT section_name, text FROM chunks WHERE paper_id = ? ORDER BY position",
        (paper_id,),
    ).fetchall()
    if not rows:
        return ""

    def priority(name: str) -> int:
        n = (name or "").strip().lower()
        for i, key in enumerate(_SUMMARY_PRIORITY):
            if n.startswith(key):
                return i
        return len(_SUMMARY_PRIORITY)

    ordered = sorted(enumerate(rows), key=lambda t: (priority(t[1][0]), t[0]))
    out: list[str] = []
    total = 0
    for _, (_sec, txt) in ordered:
        block = (txt or "").strip()
        if not block:
            continue
        if total + len(block) > max_chars:
            block = block[: max(0, max_chars - total)]
        out.append(block)
        total += len(block)
        if total >= max_chars:
            break
    return "\n\n".join(out)


def summarize_with_llm(title: str, text: str, model: str, base_url: str = OLLAMA_BASE_URL) -> str:
    system = (
        "You are a scientific editor writing a concise, self-contained summary of a "
        "research paper for a busy researcher deciding whether to read it. "
        "Write 150-220 words of flowing prose in one or two paragraphs covering: the "
        "problem and motivation, the approach or methods, the key findings or results, "
        "and why the work matters. "
        "Do NOT output a table of contents, a list of section headings, bullet points, or "
        "markdown headers. Do not invent details that are not supported by the text. "
        "If the provided text is too garbled or sparse to summarize, reply with exactly: "
        "INSUFFICIENT_TEXT"
    )
    user = f"Title: {title}\n\nPaper excerpts:\n{text}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.9, "num_ctx": 8192},
    }
    apply_thinking_off(payload, model)
    r = requests.post(f"{base_url}/api/chat", json=payload, timeout=300)
    r.raise_for_status()
    return strip_think(r.json().get("message", {}).get("content") or "")


def summarize_paper(
    conn: sqlite3.Connection,
    paper_id: str,
    model: str,
    base_url: str = OLLAMA_BASE_URL,
    force: bool = False,
) -> str:
    """Return a cached summary, generating and caching one if absent. Returns ""
    when there's too little usable text or no model was supplied."""
    ensure_summary_col(conn)
    if not force:
        row = conn.execute("SELECT summary FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        if row and row[0]:
            return row[0]

    trow = conn.execute("SELECT title FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    if not trow:
        return ""
    title = trow[0] or paper_id

    text = gather_paper_text(conn, paper_id)
    if len(text) < 200:
        return ""
    if not model:
        return ""

    summary = summarize_with_llm(title, text, model, base_url)
    if not summary or summary.strip() == "INSUFFICIENT_TEXT":
        return ""

    conn.execute("UPDATE papers SET summary = ? WHERE paper_id = ?", (summary, paper_id))
    conn.commit()
    return summary
