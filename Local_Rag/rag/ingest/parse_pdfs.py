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

# ── Garbled-text detection (broken PDF font encoding) ─────────────────────────
# Some PDFs embed fonts with a missing/broken ToUnicode map. The page renders the
# right shapes, but text extraction yields glyph soup: raw Adobe glyph names like
# "/a114" and decorative Unicode (dingbats, geometric shapes, math-alphanumerics)
# standing in for letters. We detect that and re-parse the PDF with forced OCR.

# Unicode blocks that are normal *symbols* but become noise when a font (mis)uses
# them as letter glyphs.
_SUS_RANGES = (
    (0x2190, 0x21FF),    # Arrows
    (0x2460, 0x24FF),    # Enclosed Alphanumerics
    (0x2500, 0x257F),    # Box Drawing
    (0x2580, 0x25FF),    # Block Elements + Geometric Shapes (▼ ▲ ● ◆ …)
    (0x2600, 0x26FF),    # Miscellaneous Symbols (♠ ♥ ☀ …)
    (0x2700, 0x27BF),    # Dingbats (❈ ❍ ❆ ❯ ✉ ✐ …)
    (0x1D400, 0x1D7FF),  # Mathematical Alphanumeric Symbols
    (0x1F100, 0x1F1FF),  # Enclosed Alphanumeric Supplement
    (0xE000, 0xF8FF),    # Private Use Area
)

# Raw glyph-name tokens that leak into extracted text, e.g. "/a114", "/g3", "/uni0041".
_GLYPH_TOKEN_RE = re.compile(r"/(?:uni[0-9A-Fa-f]{4}|[A-Za-z]{1,4}\d{1,4})")


def _is_sus_char(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _SUS_RANGES)


def garbage_ratio(text: str) -> float:
    """Fraction of meaningful glyphs that are unreadable (0.0 clean … 1.0 garbage)."""
    if not text:
        return 0.0
    glyph_tokens = _GLYPH_TOKEN_RE.findall(text)
    stripped = _GLYPH_TOKEN_RE.sub(" ", text)
    good = sus = 0
    for ch in stripped:
        if _is_sus_char(ch):
            sus += 1
        elif ch.isalnum():
            good += 1
    bad = sus + len(glyph_tokens)
    total = good + bad
    return bad / total if total else 0.0


def _sections_text(sections: list[dict], title: str | None = None, cap: int = 40000) -> str:
    parts = [title or ""]
    parts.extend(s.get("text", "") for s in (sections or []))
    return " ".join(p for p in parts if p)[:cap]


def is_garbled(sections: list[dict], title: str | None = None,
               *, min_chars: int = 80, threshold: float = 0.25) -> bool:
    """True when extracted text is dominated by glyph soup and worth re-OCRing."""
    sample = _sections_text(sections, title)
    if len(sample) < min_chars:
        return False
    return garbage_ratio(sample) >= threshold


_ocr_converter: DocumentConverter | None = None
_ocr_unavailable = False


def _get_ocr_converter() -> DocumentConverter | None:
    """Lazily build a DocumentConverter that forces full-page OCR, bypassing the
    PDF's (broken) embedded text layer. Returns None if OCR can't be configured."""
    global _ocr_converter, _ocr_unavailable
    if _ocr_converter is not None:
        return _ocr_converter
    if _ocr_unavailable:
        return None
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption

        opts = PdfPipelineOptions()
        opts.do_ocr = True
        # Re-OCR the whole page instead of trusting the embedded text layer.
        try:
            opts.ocr_options.force_full_page_ocr = True
        except Exception:
            pass
        _ocr_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        return _ocr_converter
    except Exception as e:  # OCR engine/deps missing — degrade gracefully.
        log.warning("Full-page OCR unavailable (%s); keeping best-effort text.", e)
        _ocr_unavailable = True
        return None


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
    if len(t) < 6 or bool(_JUNK_TITLE_RE.search(t)):
        return True
    # Glyph-soup title from a broken font encoding → treat as junk.
    return garbage_ratio(t) >= 0.25


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


def _extract_with_fallback(doc) -> tuple[str | None, list[dict], str]:
    """Structured extraction, falling back to flat markdown if it yields nothing."""
    doc_title, body_sections, refs_raw = _extract_structured(doc)
    if not body_sections:
        try:
            md = doc.export_to_markdown()
        except Exception:
            md = ""
        if md.strip():
            body_sections = [{"name": "Body", "text": md.strip(),
                              "page_start": 1, "page_end": 1}]
    return doc_title, body_sections, refs_raw


def _convert(converter: DocumentConverter, pdf_path: Path):
    try:
        return converter.convert(str(pdf_path)).document
    except Exception as e:
        log.error("Docling failed for %s: %s", pdf_path.name, e)
        return None


def parse_pdf(pdf_path: Path, library: dict, converter: DocumentConverter) -> dict | None:
    paper_id = _slug(pdf_path)
    stem = pdf_path.stem

    doc = _convert(converter, pdf_path)
    if doc is None:
        return None

    doc_title, body_sections, refs_raw = _extract_with_fallback(doc)

    # Broken font encoding → glyph soup. Re-parse this PDF with forced full-page
    # OCR and keep whichever extraction is cleaner.
    if is_garbled(body_sections, doc_title):
        ocr_conv = _get_ocr_converter()
        if ocr_conv is not None:
            log.warning("Garbled text in %s (ratio %.2f) — retrying with full-page OCR…",
                        pdf_path.name, garbage_ratio(_sections_text(body_sections, doc_title)))
            ocr_doc = _convert(ocr_conv, pdf_path)
            if ocr_doc is not None:
                o_title, o_sections, o_refs = _extract_with_fallback(ocr_doc)
                if o_sections and garbage_ratio(_sections_text(o_sections, o_title)) < \
                        garbage_ratio(_sections_text(body_sections, doc_title)):
                    doc_title = o_title or doc_title
                    body_sections = o_sections
                    refs_raw = o_refs or refs_raw
                    log.info("OCR recovered readable text for %s", pdf_path.name)

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


def run(limit: int | None = None, force: bool = False) -> None:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    library = _load_library()
    converter = DocumentConverter()

    # Case-insensitive match (.pdf/.PDF) — glob('*.pdf') would miss `Foo.PDF`.
    pdfs = sorted(
        p for p in PAPERS_PDF_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf" and not p.name.startswith("._")
    )
    if limit:
        pdfs = pdfs[:limit]

    skipped, ok = [], 0
    for pdf_path in pdfs:
        out_path = PARSED_DIR / f"{_slug(pdf_path)}.json"
        if out_path.exists() and not force:
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
    parser.add_argument("--force", action="store_true",
                        help="Re-parse even if a cached JSON already exists")
    args = parser.parse_args()
    run(args.limit, force=args.force)
