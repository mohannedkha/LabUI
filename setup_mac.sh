#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# One-shot macOS setup for LabUI: build the venv (Apple-silicon/MPS wheels),
# verify the sqlite-vec extension loads, and pull the Ollama model.
#
# Run from the project root ON THE MAC, after copying the code + data over:
#   ./setup_mac.sh
# Then:
#   ./start.sh
#
# Idempotent: re-running reuses an existing venv (rm -rf submission_strategy/.venv
# to rebuild clean). Override the model with CODEX_GEN_MODEL=...
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

APPS_ROOT="$(cd "$(dirname "$0")" && pwd)"
JFR_ROOT="$APPS_ROOT/submission_strategy"
RAG_REQS="$APPS_ROOT/Local_Rag/rag/requirements.txt"
VENV="$JFR_ROOT/.venv"
GEN_MODEL="${RAG_GEN_MODEL:-${CODEX_GEN_MODEL:-}}"   # empty = auto-detect from Ollama
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

echo "════════════════════════════════════════════════"
echo "  LabUI — macOS setup"
echo "════════════════════════════════════════════════"

# ── 1. Locate a non-Apple Python 3.12 ────────────────────────────────────────
# Apple's /usr/bin/python3 disables sqlite extension loading, which sqlite-vec
# needs — so we require a Homebrew / python.org interpreter.
PY=""
for c in /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12 python3.12; do
  if command -v "$c" >/dev/null 2>&1; then PY="$(command -v "$c")"; break; fi
done
if [ -z "$PY" ]; then
  echo "ERROR: python3.12 not found."
  echo "       Install it with:  brew install python@3.12"
  exit 1
fi
case "$PY" in
  /usr/bin/*)
    echo "ERROR: $PY is Apple's system Python — it blocks sqlite extension loading."
    echo "       Install Homebrew Python:  brew install python@3.12"
    exit 1 ;;
esac
echo "→ Python: $PY ($("$PY" --version 2>&1))"

# ── 2. Build / reuse the venv ─────────────────────────────────────────────────
# A venv copied from another machine (or carried across an OS reinstall) is
# broken: its interpreter symlinks dangle, so pip falls back to the
# externally-managed Homebrew Python and aborts with
# "error: externally-managed-environment". Validate before reusing; rebuild if
# the interpreter is missing or not the Python 3.12 we expect.
venv_ok() {
  [ -x "$VENV/bin/python" ] && \
    "$VENV/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)' >/dev/null 2>&1
}
if [ -d "$VENV" ]; then
  if venv_ok; then
    echo "→ venv exists at $VENV — reusing (rm -rf it to rebuild clean)."
  else
    echo "→ Existing venv at $VENV is stale/broken (copied from another machine or"
    echo "  a previous OS install) — rebuilding clean."
    rm -rf "$VENV"
    "$PY" -m venv "$VENV"
  fi
else
  echo "→ Creating venv…"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
# Install via the venv's interpreter explicitly so we never fall through to the
# externally-managed system Python even if activation is shadowed.
"$VENV/bin/python" -m pip install -U pip wheel >/dev/null
echo "→ Installing jfr app (editable)…"
"$VENV/bin/python" -m pip install -e "$JFR_ROOT"
echo "→ Installing RAG requirements…"
"$VENV/bin/python" -m pip install -r "$RAG_REQS"

# ── 3. Verify sqlite-vec loads (the macOS gotcha) ─────────────────────────────
echo -n "→ sqlite-vec extension: "
python - <<'PY'
import sqlite3, sqlite_vec
c = sqlite3.connect(":memory:")
c.enable_load_extension(True)
sqlite_vec.load(c)
print("OK (vec_version", c.execute("select vec_version()").fetchone()[0] + ")")
PY

# ── 4. Ollama + generator model ───────────────────────────────────────────────
# No model is hardcoded. If the user pinned one (RAG_GEN_MODEL/CODEX_GEN_MODEL),
# pull it; otherwise just confirm Ollama has at least one model to auto-detect.
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
    echo "→ Ollama has no models yet. Pull one, e.g.:  ollama pull llama3.1:8b"
  fi
else
  echo "⚠ Ollama not reachable at ${OLLAMA_URL}."
  echo "  Start it (open Ollama.app or run 'ollama serve'), then pull a model, e.g.:"
  echo "    ollama pull llama3.1:8b"
fi

echo
echo "✓ Setup complete. Start the server with:  ./start.sh"
echo "  (First server run also downloads bge-m3 + reranker from HuggingFace, ~2.5 GB.)"
