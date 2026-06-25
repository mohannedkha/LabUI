#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# One-shot setup for LabUI: build an isolated venv (Apple-silicon/MPS wheels),
# install all dependencies, verify the sqlite-vec extension loads, and check
# Ollama. Designed to run on a fresh machine with only the minimum installed.
#
# Run from the project root:
#   ./setup_mac.sh
# Then:
#   ./start.sh
#
# Idempotent and self-healing: a stale/copied venv is detected and rebuilt.
# Override the model with RAG_GEN_MODEL=… (else the app auto-detects from Ollama).
# ─────────────────────────────────────────────────────────────────────────────

# Re-exec under bash if started with `sh setup_mac.sh` (we use bash-only features
# like `set -o pipefail`). Without this, `sh` users get cryptic failures.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

APPS_ROOT="$(cd "$(dirname "$0")" && pwd)"
JFR_ROOT="$APPS_ROOT/submission_strategy"
ROOT_REQS="$APPS_ROOT/requirements.txt"            # consolidated single source of truth
RAG_REQS="$APPS_ROOT/Local_Rag/rag/requirements.txt"
VENV="$JFR_ROOT/.venv"
PYBIN="$VENV/bin/python"   # always install/run through THIS, never a bare `pip`
GEN_MODEL="${RAG_GEN_MODEL:-${LABUI_GEN_MODEL:-${CODEX_GEN_MODEL:-}}}"  # empty = auto-detect
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

echo "════════════════════════════════════════════════"
echo "  LabUI — setup"
echo "════════════════════════════════════════════════"

# ── 1. Locate a usable Python (3.12 preferred) ───────────────────────────────
# Must NOT be Apple's /usr/bin/python3: it disables SQLite extension loading,
# which sqlite-vec (the vector index) needs. We want Homebrew / python.org /
# pyenv. Prefer 3.12 (what the wheels are tested against); accept 3.11/3.13.
find_python() {
  local ver cand resolved
  for ver in 3.12 3.11 3.13; do
    for cand in \
      "/opt/homebrew/bin/python$ver" \
      "/usr/local/bin/python$ver" \
      "/Library/Frameworks/Python.framework/Versions/$ver/bin/python$ver" \
      "python$ver"; do
      if command -v "$cand" >/dev/null 2>&1; then
        resolved="$(command -v "$cand")"
        case "$(cd "$(dirname "$resolved")" && pwd)/$(basename "$resolved")" in
          /usr/bin/*|/System/*) continue ;;   # skip Apple system Python
        esac
        echo "$resolved"; return 0
      fi
    done
  done
  return 1
}

PY="$(find_python || true)"

# Nothing suitable found — try to install it automatically if Homebrew is here.
if [ -z "$PY" ]; then
  if command -v brew >/dev/null 2>&1; then
    echo "→ No suitable Python found — installing python@3.12 via Homebrew…"
    brew install python@3.12
    PY="$(find_python || true)"
  fi
fi
if [ -z "$PY" ]; then
  echo "ERROR: need a non-Apple Python 3.12 (3.11/3.13 also accepted)." >&2
  echo "       Install Homebrew (https://brew.sh), then:  brew install python@3.12" >&2
  exit 1
fi
echo "→ Python: $PY ($("$PY" --version 2>&1))"
case "$("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])')" in
  3.12) ;;
  *) echo "  (note: 3.12 is the tested version; continuing on $("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])') — some wheels may differ.)" ;;
esac

# ── 2. Build / reuse the venv ─────────────────────────────────────────────────
# A venv copied between machines or carried across an OS reinstall is broken:
# its interpreter symlinks dangle, so `pip` falls back to the externally-managed
# system Python and aborts ("error: externally-managed-environment"). Reuse only
# a venv whose interpreter actually runs AND has a working pip; otherwise rebuild.
venv_ok() {
  [ -x "$PYBIN" ] \
    && "$PYBIN" -c 'import sys' >/dev/null 2>&1 \
    && "$PYBIN" -m pip --version >/dev/null 2>&1
}
if [ -d "$VENV" ] && venv_ok; then
  echo "→ venv exists and is healthy at $VENV — reusing."
else
  if [ -d "$VENV" ]; then
    echo "→ Existing venv at $VENV is stale/broken (copied from another machine, a"
    echo "  previous OS install, or a partial setup) — rebuilding clean."
    rm -rf "$VENV"
  else
    echo "→ Creating venv at $VENV…"
  fi
  "$PY" -m venv "$VENV"
fi

# Install EVERYTHING through the venv's own interpreter. This is the whole game:
# `$PYBIN -m pip` can never hit externally-managed, regardless of PATH/activation.
echo "→ Upgrading pip/wheel…"
"$PYBIN" -m pip install -U pip wheel >/dev/null
echo "→ Installing jfr app (editable)…"
"$PYBIN" -m pip install -e "$JFR_ROOT"
echo "→ Installing RAG requirements (torch, docling, … — this is the slow part)…"
"$PYBIN" -m pip install -r "$RAG_REQS"
# The consolidated top-level list is the single source of truth for the whole app
# (PyMuPDF for PDF reading, python-multipart for uploads, httpx, feedparser, …).
# Install it last so its known-good version floors win.
if [ -f "$ROOT_REQS" ]; then
  echo "→ Installing consolidated app requirements (PDF reading, uploads, web, …)…"
  "$PYBIN" -m pip install -r "$ROOT_REQS"
fi

# ── 3. Verify the install really completed (fail loudly HERE, not at runtime) ──
echo "→ Verifying sqlite-vec + core imports…"
"$PYBIN" - <<'PY'
import importlib, sqlite3, sys

# Functional check: the SQLite vector extension must load (the macOS gotcha).
import sqlite_vec
c = sqlite3.connect(":memory:")
c.enable_load_extension(True)
sqlite_vec.load(c)
ver = c.execute("select vec_version()").fetchone()[0]

# Import check: every module the app needs at runtime. A partial `pip install`
# (one bad wheel can abort the whole thing) is caught here instead of on boot.
required = [
    "fastapi", "uvicorn", "qdrant_client", "jinja2",
    "torch", "sentence_transformers", "transformers",
    "docling", "rank_bm25", "numpy", "pydantic", "requests",
    "fitz",       # PyMuPDF — PDF text extraction (chat attachments / reading)
    "feedparser", "apscheduler",  # journal corpus refresh + scheduler
]
missing = []
for m in required:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(f"{m} ({type(e).__name__}: {e})")

if missing:
    print("  ✗ INCOMPLETE install — these failed to import:", file=sys.stderr)
    for m in missing:
        print("      -", m, file=sys.stderr)
    print("  Re-run ./setup_mac.sh; if one package keeps failing to build, install", file=sys.stderr)
    print("  Xcode command-line tools (xcode-select --install) and try again.", file=sys.stderr)
    sys.exit(1)

print(f"  ✓ sqlite-vec OK (vec_version {ver}); all {len(required)} core modules import.")
PY

# ── 4. Ollama + generator model ───────────────────────────────────────────────
# No model is hardcoded. If one is pinned, pull it; otherwise confirm Ollama has
# at least one model to auto-detect (the app prefers Qwen3 — see RAG_GEN_PREFER).
if curl -fs --max-time 3 "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
  TAGS_JSON="$(curl -fs "${OLLAMA_URL}/api/tags")"
  if [ -n "${GEN_MODEL}" ]; then
    if printf '%s' "$TAGS_JSON" | grep -q "\"${GEN_MODEL}\""; then
      echo "→ Pinned model ${GEN_MODEL} already present."
    else
      echo "→ Pulling pinned model ${GEN_MODEL} (one-time download)…"
      ollama pull "${GEN_MODEL}"
    fi
  elif printf '%s' "$TAGS_JSON" | grep -q '"name"'; then
    HAVE="$(printf '%s' "$TAGS_JSON" | grep -o '"name":"[^"]*"' | head -3 | sed 's/.*:"//;s/"//' | paste -sd, -)"
    echo "→ No model pinned; the app will auto-detect from Ollama (have: ${HAVE})."
  else
    echo "→ Ollama has no models yet. Pull one, e.g.:  ollama pull qwen3:8b"
  fi
else
  echo "⚠ Ollama not reachable at ${OLLAMA_URL} (this is fine for setup)."
  echo "  Before running the app: install Ollama (https://ollama.com), then:"
  echo "    ollama serve            # or open Ollama.app"
  echo "    ollama pull qwen3:8b    # any model — larger = better quality"
fi

echo
echo "✓ Setup complete. Start the server with:  ./start.sh"
echo "  (First server run also downloads bge-m3 + reranker from HuggingFace, ~2.5 GB.)"
