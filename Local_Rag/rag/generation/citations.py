"""
ACS-style numbered citations.

One number per *paper* (not per chunk), assigned in order of first appearance in
the retrieved chunk list. The same map drives three places so they stay aligned:
  • the prompt (each excerpt is labelled [n] and the model cites [n] inline),
  • the source panel in the UI,
  • the code-built "References" list appended after generation.

Metadata is thin (authors are often the literal "See paper", year is often
NULL), so references are built from the reliable title field and only include
authors/year when they are real.
"""
from __future__ import annotations

import json
import re

# Author strings that are placeholders, not real author lists.
_AUTHOR_PLACEHOLDERS = {"see paper", "unknown", "n/a", "na", "none", ""}


def build_citation_map(chunks: list[dict]) -> dict[str, int]:
    """paper_id -> citation number (1-based), by order of first appearance."""
    mapping: dict[str, int] = {}
    for c in chunks:
        pid = c.get("paper_id")
        if pid and pid not in mapping:
            mapping[pid] = len(mapping) + 1
    return mapping


def _first_meta_by_paper(chunks: list[dict]) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for c in chunks:
        pid = c.get("paper_id")
        if pid and pid not in meta:
            meta[pid] = c
    return meta


def _clean_authors(authors) -> str:
    """Normalise the authors field; '' if it is missing or a placeholder."""
    if not authors:
        return ""
    s = authors if isinstance(authors, str) else str(authors)
    s = s.strip()
    if s.lower() in _AUTHOR_PLACEHOLDERS:
        return ""
    # Stored as a JSON list in some rows.
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            s = "; ".join(str(x).strip() for x in parsed if str(x).strip())
    except Exception:
        pass
    # Drop affiliation markers like "Name1,2*" -> "Name", then heal the commas
    # those superscripts leave behind ("Ohmori,, Haruta" -> "Ohmori, Haruta").
    s = re.sub(r"[\d\*†‡§]+", "", s)
    s = re.sub(r"(\s*,\s*){2,}", ", ", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" ;,")
    return s


def format_reference(n: int, c: dict) -> str:
    """One reference line: '[n] Authors. Title. Year.' (parts omitted if absent)."""
    title = (c.get("title") or c.get("paper_id") or "").strip().rstrip(".")
    authors = _clean_authors(c.get("authors"))
    year = c.get("year")
    parts: list[str] = []
    if authors:
        parts.append(authors)
    if title:
        parts.append(title)
    if year:
        parts.append(str(year))
    body = ". ".join(parts) if parts else (c.get("paper_id") or "untitled")
    line = f"[{n}] {body}."
    doi = (c.get("doi") or "").strip()
    if doi:
        line += f" https://doi.org/{doi}"
    return line


def extract_cited_numbers(text: str, max_n: int) -> set[int]:
    """All citation numbers actually used in the text, within [1, max_n]."""
    nums: set[int] = set()
    for grp in re.findall(r"\[([0-9,\s–\-]+)\]", text):
        for part in re.split(r"[,\s]+", grp.strip()):
            if not part:
                continue
            rng = re.match(r"^(\d+)[–\-](\d+)$", part)
            if rng:
                for k in range(int(rng.group(1)), int(rng.group(2)) + 1):
                    nums.add(k)
            elif part.isdigit():
                nums.add(int(part))
    return {n for n in nums if 1 <= n <= max_n}


# A trailing reference/bibliography block the model sometimes writes despite
# being told not to. Matched as a standalone heading line near the end of the
# answer. "Citations"/"Sources" are deliberately excluded — some agents produce
# those as legitimate primary output.
_MODEL_REF_HEADING_RE = re.compile(
    r"\n[ \t]*(?:[-*_]{3,}[ \t]*\n[ \t]*)?"          # optional --- rule line
    r"(?:#{1,6}[ \t]*)?\**[ \t]*"                     # optional ##/** markers
    r"(?:references|reference list|cited papers|cited works|works cited|"
    r"bibliography)\b[ \t:]*\**[ \t]*\n",
    re.IGNORECASE,
)


def strip_model_references(text: str) -> str:
    """Remove a reference/bibliography block the model appended on its own."""
    matches = list(_MODEL_REF_HEADING_RE.finditer(text))
    if not matches:
        return text
    cut = matches[-1].start()
    # Only strip when it sits in the tail, so a genuine mid-answer mention is safe.
    if cut < len(text) * 0.4:
        return text
    return text[:cut].rstrip().rstrip("-—_*").rstrip()


def build_references(chunks: list[dict], text: str) -> str:
    """
    Markdown "## References" block listing only the sources actually cited in
    `text`, in ascending number order. Returns "" if nothing was cited.
    """
    mapping = build_citation_map(chunks)
    if not mapping:
        return ""
    cited = extract_cited_numbers(text, len(mapping))
    if not cited:
        return ""
    meta = _first_meta_by_paper(chunks)
    inv = {n: pid for pid, n in mapping.items()}
    lines = [format_reference(n, meta[inv[n]]) for n in sorted(cited)]
    return "\n\n## References\n\n" + "\n\n".join(lines)
