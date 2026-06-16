#!/usr/bin/env python3
"""
Stage 1 — Parse PDFs into structured JSON using Docling.
Run: python3 -m ingest.parse_pdfs [--limit N]

Builds sections from Docling's structured document items (not flat markdown) so
each section carries REAL page numbers (from item provenance) and a real title.
Running page headers/footers are dropped, and reference/back-matter sections are
routed to references_raw instead of being indexed as retrievable chunks.
"""
import argparse
import json
import logging
import re
import sys
from pathlib import Path

try:
    from docling.document_converter import DocumentConverter
except ImportError:
    print("Error: docling is not installed. Please run: pip install docling")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PAPERS_PDF_DIR, PAPERS_LIBRARY, PARSED_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logging.getLogger("RapidOCR").setLevel(logging.ERROR)
log = logging.getLogger(__name__)

# Section names that are back-matter / non-content → routed to references_raw,
# never indexed as retrievable chunks.
_REF_HEAD_RE = re.compile(
    r"^\s*(references?|bibliography|acknowledg(e?ment)?s?|"
    r"supporting\s+information|supplementary(\s+(materials?|information|data))?|"
    r"conflicts?\s+of\s+interest|declaration\s+of\s+competing\s+interest|"
    r"author\s+contributions?|funding|data\s+availability|"
    r"references\s+and\s+notes)\b",
    re.IGNORECASE,
)

# Docling item labels whose text we treat as body content.
_CONTENT_LABELS = {"text", "paragraph", "list_item", "formula", "caption", "code"}
# Labels to ignore entirely (running heads/feet, page furniture, figures/tables).
_SKIP_LABELS = {"page_header", "page_footer", "picture", "table", "footnote"}

_JUNK_TITLE_RE = re.compile(
    r"(view\s+article\s+online|^\s*$|licensed\s+under|downloaded|"
    r"^\s*issn|doi\.org|http|this\s+article|^\s*see\s+paper\s*$|"
    r"is\s+an?\s+international\s+journal|^\s*proceedings\b|repository)",
    re.IGNORECASE,
)


def _slug(path: Path) -> str:
    name = path.stem
    name = re.sub(r"[^\w\-]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:120]


def _load_library() -> dict:
    if not PAPERS_LIBRARY.exists():
        log.warning("papers_library.json not found at %s", PAPERS_LIBRARY)
        return {}
    with open(PAPERS_LIBRARY, encoding="utf-8") as f:
        raw = json.load(f)
    lib = {}
    for entry in (raw if isinstance(raw, list) else raw.get("papers", [])):
        src = entry.get("filename") or entry.get("source") or entry.get("file") or ""
        stem = Path(src).stem if src else ""
        if stem:
            lib[stem] = entry
    return lib


def _label(item) -> str:
    lab = getattr(item, "label", None)
    if lab is None:
        return ""
    return str(getattr(lab, "value", lab)).lower()


def _page_of(item) -> int | None:
    prov = getattr(item, "prov", None)
    if prov:
        pno = getattr(prov[0], "page_no", None)
        if isinstance(pno, int):
            return pno
    return None


def _is_junk_title(t: str) -> bool:
    t = (t or "").strip()
    return len(t) < 6 or bool(_JUNK_TITLE_RE.search(t))


def _extract_structured(doc) -> tuple[str | None, list[dict], str]:
    """Return (title, body_sections, references_raw) from a DoclingDocument."""
    doc_title: str | None = None
    sections: list[dict] = []
    refs_parts: list[str] = []
    cur: dict | None = None
    last_page = 1

    def flush(section: dict | None) -> None:
        if section is None:
            return
        text = "\n".join(section["parts"]).strip()
        if not text:
            return
        if _REF_HEAD_RE.match(section["name"] or ""):
            refs_parts.append(text)
        else:
            sections.append({
                "name": section["name"] or "Body",
                "text": text,
                "page_start": section["page_start"],
                "page_end": section["page_end"],
            })

    for item, _level in doc.iterate_items():
        labv = _label(item)
        if labv in _SKIP_LABELS:
            continue
        text = (getattr(item, "text", "") or "").strip()
        page = _page_of(item) or last_page
        last_page = page

        if labv == "title":
            if not doc_title and text:
                doc_title = text
            continue

        if labv in ("section_header", "section-header"):
            flush(cur)
            cur = {"name": text or "Body", "parts": [],
                   "page_start": page, "page_end": page}
            continue

        if labv in _CONTENT_LABELS and text:
            if cur is None:
                cur = {"name": "Body", "parts": [],
                       "page_start": page, "page_end": page}
            cur["parts"].append(text)
            cur["page_end"] = page

    flush(cur)
    return doc_title, sections, "\n".join(refs_parts).strip()


def parse_pdf(pdf_path: Path, library: dict, converter: DocumentConverter) -> dict | None:
    paper_id = _slug(pdf_path)
    stem = pdf_path.stem

    try:
        result = converter.convert(str(pdf_path))
        doc = result.document
    except Exception as e:
        log.error("Docling failed for %s: %s", pdf_path.name, e)
        return None

    doc_title, body_sections, refs_raw = _extract_structured(doc)

    # Fallback: if structured extraction produced nothing, use flat markdown.
    if not body_sections:
        try:
            md = doc.export_to_markdown()
        except Exception:
            md = ""
        if md.strip():
            body_sections = [{"name": "Body", "text": md.strip(),
                              "page_start": 1, "page_end": 1}]

    # --- Title: prefer Docling's detected title, then library, then filename ---
    lib_entry = library.get(stem, {})
    lib_title = lib_entry.get("title")
    if doc_title and not _is_junk_title(doc_title):
        title = doc_title.strip()
    elif lib_title and not _is_junk_title(lib_title):
        title = lib_title.strip()
    else:
        title = stem

    # --- Authors / year from library when available ---
    authors = lib_entry.get("authors", [])
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(";") if a.strip()]
    year = lib_entry.get("year") or lib_entry.get("date", "")
    if year:
        m = re.search(r"(19|20)\d{2}", str(year))
        year = int(m.group()) if m else None
    else:
        year = None

    return {
        "paper_id": paper_id,
        "title": title,
        "authors": authors,
        "year": year,
        "source_pdf": str(pdf_path),
        "sections": body_sections,
        "references_raw": refs_raw,
    }


def run(limit: int | None = None) -> None:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    library = _load_library()
    converter = DocumentConverter()

    pdfs = sorted(PAPERS_PDF_DIR.glob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]

    skipped, ok = [], 0
    for pdf_path in pdfs:
        out_path = PARSED_DIR / f"{_slug(pdf_path)}.json"
        if out_path.exists():
            ok += 1
            continue

        result = parse_pdf(pdf_path, library, converter)
        if result is None:
            skipped.append(pdf_path.name)
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        ok += 1
        log.info("Parsed [%d/%d] %s", ok, len(pdfs), pdf_path.name)

    log.info("Done. %d parsed, %d skipped.", ok, len(skipped))
    if skipped:
        log.warning("Skipped PDFs: %s", ", ".join(skipped))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.limit)
