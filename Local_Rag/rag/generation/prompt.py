"""
Citation-enforced prompt builder for Ollama chat API.
Supports per-agent system prompts via generation.agents.
Supports doc_context for uploaded PDF/document text.
"""
from generation.agents import get_agent, AGENTS, DEFAULT_AGENT
from generation.citations import build_citation_map


def build_user_prompt(
    query: str,
    chunks: list[dict],
    agent_id: str = DEFAULT_AGENT,
    web_results: list[dict] | None = None,
    doc_context: str | None = None,
    memories: list[dict] | None = None,
    style_samples: str | None = None,
) -> str:
    agent = get_agent(agent_id)

    # ── Local paper excerpts ──────────────────────────────────────────────────
    # One citation number per paper, shared with the source panel and the
    # auto-appended References list (see generation.citations).
    cite_map = build_citation_map(chunks)
    excerpt_lines = []
    for c in chunks:
        n = cite_map.get(c["paper_id"], "?")
        block = (
            f"---\n"
            f"[{n}] {c.get('title', '')}\n"
            f"Section: {c.get('section_name', '')} "
            f"(pages {c.get('page_start', '?')}-{c.get('page_end', '?')})\n"
            f"{c['text']}\n"
            f"---"
        )
        excerpt_lines.append(block)

    excerpts = "\n\n".join(excerpt_lines) if excerpt_lines else "(no local papers retrieved)"

    # ── Web search results ────────────────────────────────────────────────────
    web_section = ""
    if web_results:
        web_lines = []
        for r in web_results:
            web_lines.append(
                f"[web:{r['index']}] {r['title']}\n"
                f"URL: {r['url']}\n"
                f"{r['snippet']}"
            )
        web_section = (
            "\n\n# Web search results (supplementary — cite as [web:N])\n"
            + "\n\n".join(web_lines)
        )

    # ── Uploaded document context ─────────────────────────────────────────────
    doc_section = ""
    if doc_context:
        doc_section = f"\n\n# Attached document (user-uploaded)\n{doc_context}"

    # ── Long-term memory context ──────────────────────────────────────────────
    memory_section = ""
    if memories:
        mem_lines = "\n".join(f"- {m['content']}" for m in memories)
        memory_section = (
            "\n\n# Research memory (findings from past conversations — use as background context)\n"
            + mem_lines
        )

    # ── Writing-style samples (writing agent) ─────────────────────────────────
    style_section = ""
    if style_samples:
        style_section = (
            "\n\n# Writing style samples (the user's own prose — match this voice, "
            "rhythm, and vocabulary)\n"
            + style_samples
        )

    instruction = agent.get("instruction", "")
    if web_results:
        instruction += (
            " You may also cite web search results as [web:N] where N is the result number. "
            "Prefer local paper excerpts over web results when both are available."
        )
    if doc_context:
        instruction += (
            " The user has also attached a document whose full text is provided above. "
            "You may reference it directly in your answer."
        )

    return (
        f"# Input\n{query}\n\n"
        f"# Local paper excerpts\n{excerpts}"
        f"{web_section}"
        f"{doc_section}"
        f"{memory_section}"
        f"{style_section}\n\n"
        f"# Instructions\n{instruction}"
    )


def build_messages(
    query: str,
    chunks: list[dict],
    agent_id: str = DEFAULT_AGENT,
    web_results: list[dict] | None = None,
    doc_context: str | None = None,
    memories: list[dict] | None = None,
    style_samples: str | None = None,
) -> list[dict]:
    agent = get_agent(agent_id)
    return [
        {"role": "system", "content": agent["system"]},
        {"role": "user",   "content": build_user_prompt(
            query, chunks, agent_id, web_results, doc_context, memories,
            style_samples,
        )},
    ]
