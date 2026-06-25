# LabUI

A local, private research workbench: **chat with your own PDF library** (retrieval-augmented,
with citations) plus a **journal-fit recommender / submission tracker**. Everything runs on your
machine — your papers, the vector index, and the LLM (via [Ollama](https://ollama.com)) never
leave it.

This is a clean, self-contained instance: it ships with **no papers and no data**. You add your
own PDFs and they get indexed directly.

---

## What you get

- **Research chat** (`/research/chat`) — ask questions across your library; answers stream with
  inline `[n]` citations and a sources panel. Optional web-search augmentation.
- **Paper search** (`/research/search`) — hybrid BM25 + dense (bge-m3) retrieval, reranked.
- **Upload** (`/research/upload`) — drag in PDFs **or a whole folder**; they auto-index.
- **Notes** (`/research/notes`) — save answers and write-ups; one Markdown file each.
- **Markdown + LaTeX math** — chat answers, notes, and experiment results render Markdown with
  KaTeX, so inline `$E = mc^2$` and display `$$\frac{a}{b}$$` math typeset properly.
- **Journal-Fit** (`/journals`, `/manuscripts`) — match a manuscript to candidate journals.
  Ships with a starter set of 23 venues; add more from the **/journals** page by searching
  online (Crossref, by name or ISSN) or entering them manually.

## Prerequisites

- **macOS (Apple Silicon)** or Linux, 16 GB+ RAM (32 GB recommended for large contexts).
- **Python 3.12** from Homebrew — *not* Apple's `/usr/bin/python3` (it blocks the SQLite
  extension the vector index needs): `brew install python@3.12`
- **[Ollama](https://ollama.com)** running, with at least one model pulled:
  ```bash
  ollama serve                 # or launch Ollama.app
  ollama pull llama3.1:8b      # any model — the app auto-detects what you have
  ```

## Setup

```bash
cd LabUI
./setup_mac.sh        # builds the venv, verifies sqlite-vec, checks Ollama
```

Then start it:

```bash
./start.sh            # serves http://127.0.0.1:8770
```

First launch downloads the embedder (`bge-m3`) and reranker (~2.5 GB) from HuggingFace — once,
then cached. Open **http://127.0.0.1:8770**.

## Adding your papers

Three ways, all of which trigger automatic indexing (`parse → chunk → embed → index`):

1. **Upload page** — `/research/upload`: drag PDFs or a folder, or click *Choose files* /
   *Choose folder*. Folders are scanned recursively.
2. **Drop into the folder** — copy PDFs into `Local_Rag/papers/`. The auto-index daemon polls
   every `RAG_INGEST_POLL_INTERVAL` seconds (default 120) and picks up new files.
3. **Index now** — the button on the upload page kicks an immediate index run instead of waiting.

The first batch on a fresh instance creates the index from scratch. Watch progress in the
upload page's *Auto-index* panel, or via `GET /api/rag/status`.

> **First index is slower.** On the very first run, `docling` downloads its PDF layout models
> (~hundreds of MB, one-time) before parsing. Subsequent indexing is much faster.

## Configuration 

| Variable | Default | Purpose |
|---|---|---|
| `RAG_GEN_MODEL` | *(auto)* | Pin a generator model. Empty = auto-detect the first installed Ollama model (preferring `RAG_GEN_PREFER`). |
| `RAG_GEN_PREFER` | `qwen3,qwen2.5,qwen,llama,gemma,mistral,phi` | Substring preference order when auto-selecting (Qwen3 first). |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint. |
| `RAG_EMBED_MODEL` / `RAG_EMBED_DIM` | `BAAI/bge-m3` / `1024` | Embedder. **Changing the dim requires a fresh re-index** (it's baked into the vector table). |
| `RAG_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker. |
| `RAG_DATA_DIR` / `RAG_PAPERS_DIR` | in-tree | Relocate the index / papers folder. |
| `RAG_GEN_NUM_CTX` | `32768` | Ollama context window (lower to `16384` to save RAM). |
| `RAG_INGEST_POLL_INTERVAL` | `120` | Auto-index poll seconds. |
| `JFR_DATA_DIR` | `./.data/jfr` | Journal-Fit data (db, vectors) — kept inside LabUI by `start.sh`. |
| `LABUI_HOST` / `LABUI_PORT` | `127.0.0.1` / `8770` | Bind address/port. Set host `0.0.0.0` to expose on your LAN (no auth — do this only on a trusted network). |

The generator model can also be picked live from the in-app model dropdown (top bar), which
lists whatever you have in Ollama.

## How it works

```
LabUI/
├── start.sh / setup_mac.sh        # launcher + one-shot setup (cross-platform)
├── requirements.txt               # consolidated deps
├── submission_strategy/           # FastAPI host app (Journal-Fit) + web/ templates
│   └── jfr/web/rag_routes.py       #   RAG API mounted at /api/rag/*
└── Local_Rag/
    ├── papers/                    # YOUR PDFs land here (starts empty)
    └── rag/                       # RAG pipeline: ingest/, retrieval/, generation/
        └── data/                  # rag.db (FTS5 + sqlite-vec), created on first index
```

- **Generation**: Ollama native `/api/chat` (honors `num_ctx`), model auto-detected.
- **Embeddings/rerank**: `sentence-transformers` on MPS/CUDA/CPU (auto-detected).
- **Index**: one SQLite DB with FTS5 (BM25) + a `vec0` virtual table for dense vectors.

## Troubleshooting

- **"Ollama has no models installed"** — `ollama pull llama3.1:8b` (or any model).
- **sqlite-vec fails to load** — you're on Apple's system Python; use Homebrew `python@3.12`.
- **Uploads don't index** — check `GET /api/rag/status` → `ingest.message`; ensure PDFs are
  real (not scanned-image-only, which docling may skip) and under 200 MB.
- **Out of memory during chat** — lower `RAG_GEN_NUM_CTX=16384`, or pick a smaller Ollama model.
- **Port in use** — `LABUI_PORT=8771 ./start.sh`.

## Privacy

Fully local. The only outbound calls are: HuggingFace (one-time model download), Crossref (only
when you search for a journal to add on the /journals page), and DuckDuckGo (only if you use
in-chat web search). Disable web search by not using the Web tab.
</content>
