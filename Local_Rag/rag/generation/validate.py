"""
Citation post-processor for numbered (ACS-style) citations.

Every excerpt is labelled [1], [2], … (one number per paper). This checks that
every bracketed citation number in the response refers to a real source. Numbers
outside the valid range are flagged (not silently removed) so the user sees when
the model misbehaves. Non-numeric brackets ([ref], [web:3], plain words) are left
alone — they are legitimate placeholders, not source numbers.
"""
import re

from generation.citations import build_citation_map

# Bracket contents that are never source numbers and must not be flagged.
_NON_CITATION_RE = re.compile(
    r"""^(
        ref | refs | citation | citations | x | n | todo | author | authors |
        year | source | paper | id | web:.*
    )$""",
    re.IGNORECASE | re.VERBOSE,
)


def _bad_numbers_in_group(group: str, max_n: int) -> bool:
    """True if `group` is a numeric citation referring to a number outside 1..max_n."""
    for part in re.split(r"[,\s]+", group.strip()):
        if not part:
            continue
        rng = re.match(r"^(\d+)[–\-](\d+)$", part)
        if rng:
            a, b = int(rng.group(1)), int(rng.group(2))
            if not (1 <= a <= max_n and 1 <= b <= max_n):
                return True
        elif part.isdigit():
            if not (1 <= int(part) <= max_n):
                return True
        else:
            # Mixed/garbage content inside the bracket — not a clean number group.
            return False
    return False


def validate_citations(
    response_text: str,
    retrieved_chunks: list[dict],
    cite_required: bool = True,
) -> str:
    """
    Cross-check numbered citations against the retrieved source set. Append a
    warning block if any citation number has no corresponding source.

    cite_required=False (the prose-drafting "writing" agent) disables the check,
    since those answers carry [ref] placeholders by design.
    """
    if not cite_required:
        return response_text

    max_n = len(build_citation_map(retrieved_chunks))

    bad: set[str] = set()
    for group in re.findall(r"\[([^\[\]\n]+)\]", response_text):
        g = group.strip()
        if _NON_CITATION_RE.match(g):
            continue
        if _bad_numbers_in_group(g, max_n):
            bad.add(g)

    if bad:
        warning = (
            "\n\n---\n"
            "⚠️ Unverified citations (no matching source number): "
            + ", ".join(f"[{b}]" for b in sorted(bad))
        )
        return response_text + warning

    return response_text
