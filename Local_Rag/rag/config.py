"""Central configuration — all paths, model names, and ports.

Everything here is environment-overridable so the app ships with no machine- or
user-specific values hardcoded. Paths self-locate relative to this file; the
generator model is auto-detected from Ollama at runtime (see resolve_gen_model
in the web layer) unless RAG_GEN_MODEL pins one explicitly.
"""
import os
from pathlib import Path


def _env_path(var: str, default: Path) -> Path:
    v = os.environ.get(var)
    return Path(v).expanduser().resolve() if v else default


# ── Project root ────────────────────────────────────────────────────────────
RAG_ROOT = Path(__file__).parent.resolve()

# ── Source data (override dirs via env for a relocated/containerized install) ─
LOCAL_RAG_ROOT   = RAG_ROOT.parent
PAPERS_PDF_DIR   = _env_path("RAG_PAPERS_DIR", LOCAL_RAG_ROOT / "papers")
SUMMARIES_DIR    = LOCAL_RAG_ROOT / "output" / "individual"
PAPERS_LIBRARY   = LOCAL_RAG_ROOT / "output" / "papers_library.json"

# ── Data outputs ─────────────────────────────────────────────────────────────
DATA_DIR         = _env_path("RAG_DATA_DIR", RAG_ROOT / "data")
PARSED_DIR       = DATA_DIR / "parsed"
DB_PATH          = DATA_DIR / "rag.db"
MEMORY_DB_PATH   = DATA_DIR / "memory.db"
RERANKER_CACHE   = DATA_DIR / "reranker"

# ── Models ───────────────────────────────────────────────────────────────────
# Generator — served by Ollama. Empty string = auto-detect from the models the
# user has actually pulled (resolve_gen_model). Set RAG_GEN_MODEL to pin one.
GEN_MODEL        = (os.environ.get("RAG_GEN_MODEL")
                    or os.environ.get("LABUI_GEN_MODEL")
                    or os.environ.get("CODEX_GEN_MODEL") or "").strip()
# When auto-selecting, prefer model names containing these substrings (best first).
# Qwen3 leads: it's the highest-quality local generator we target for summaries,
# chat, and memory extraction. Override with RAG_GEN_PREFER, or pin one outright
# with RAG_GEN_MODEL / LABUI_GEN_MODEL (e.g. "qwen3:32b").
GEN_MODEL_PREFER = [s.strip() for s in os.environ.get(
    "RAG_GEN_PREFER", "qwen3,qwen2.5,qwen,llama,gemma,mistral,phi").split(",") if s.strip()]
# Back-compat alias used by a couple of status checks; derived, not authoritative.
GEN_MODEL_ALIAS  = (GEN_MODEL.split(":")[0] if GEN_MODEL else "")

# Embeddings — sentence-transformers. bge-m3: 1024-dim, 8192-token context.
# NOTE: EMBED_DIM is baked into the vector table (chunks_vec FLOAT[N]); changing
# the embedder requires a re-index. Override only on a fresh/empty index.
EMBED_MODEL_BASE = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-m3")
EMBED_ADAPTER    = ""        # unused with bge-m3; kept for import compatibility
EMBED_DIM        = int(os.environ.get("RAG_EMBED_DIM", "1024"))

# Reranker — sentence-transformers, runs on the detected device (MPS/CUDA/CPU).
RERANKER_MODEL   = os.environ.get("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL  = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Server ───────────────────────────────────────────────────────────────────
PORT             = int(os.environ.get("RAG_PORT", "5002"))
STATIC_DIR       = RAG_ROOT / "static"
NOTES_DIR        = _env_path("RAG_NOTES_DIR", RAG_ROOT / "notes")  # saved answer notes (.md)
STYLE_DIR        = DATA_DIR / "style"        # user's writing-style samples (.md/.txt)
STYLE_MAX_CHARS  = int(os.environ.get("RAG_STYLE_MAX_CHARS", "12000"))

# ── Auto-index ───────────────────────────────────────────────────────────────
INGEST_POLL_INTERVAL = int(os.environ.get("RAG_INGEST_POLL_INTERVAL", "120"))  # seconds

# ── Retrieval knobs ───────────────────────────────────────────────────────────
BM25_TOP_N       = 50
DENSE_TOP_N      = 50
RRF_K            = 60
RERANK_TOP_N     = 40   # raised to support load-more requests up to 40
FINAL_TOP_K      = 8
MAX_PER_PAPER    = 2     # diversity cap in final results
MIN_DIVERSE      = 6     # fill below this many results ignoring the cap

# ── Generation ────────────────────────────────────────────────────────────────
GEN_TEMPERATURE  = float(os.environ.get("RAG_GEN_TEMPERATURE", "0.3"))
GEN_TOP_P        = float(os.environ.get("RAG_GEN_TOP_P", "0.9"))
GEN_NUM_CTX      = int(os.environ.get("RAG_GEN_NUM_CTX", "32768"))  # Ollama num_ctx
                           # override. ~1.5 GB KV cache at 32K; set 16384 to save RAM.

# ── Ingest ───────────────────────────────────────────────────────────────────
EMBED_BATCH_SIZE = 16
CHUNK_TARGET_TOK = 800   # ~3200 chars
CHUNK_OVERLAP_TOK= 150   # ~600 chars
CHARS_PER_TOKEN  = 4     # rough estimate
