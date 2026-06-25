#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Build the whole RAG library end-to-end and leave it running:
#   parse (OCR fallback) → index → repair garbled papers → generate summaries.
#
# Usage:
#   ./build_library.sh                     # full pipeline
#   ./build_library.sh --force-summaries   # also regenerate existing summaries
#   ./build_library.sh --only-summaries    # just (re)generate summaries
#   ./build_library.sh --skip-reprocess    # skip the re-OCR repair phase
#
# Leave it running in a terminal, or background it:
#   nohup ./build_library.sh > /dev/null 2>&1 &
#   tail -f build_library.log
#
# Tip: stop the web server first (Ctrl+C in its terminal) so this job has the
# database and memory to itself — especially with a large model like Qwen3.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

APPS_ROOT="$(cd "$(dirname "$0")" && pwd)"
RAG_ROOT="$APPS_ROOT/Local_Rag/rag"
JFR_ROOT="$APPS_ROOT/submission_strategy"
VENV="$JFR_ROOT/.venv"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
LOG="$APPS_ROOT/build_library.log"

if [ ! -x "$VENV/bin/python" ]; then
  echo "ERROR: venv not found at $VENV." >&2
  echo "       Create it first:  cd $JFR_ROOT && python3.12 -m venv .venv && source .venv/bin/activate && pip install -e ." >&2
  exit 1
fi

# Summaries need Ollama; warn (don't fail) so parse/index/repair still run.
if ! curl -fs --max-time 3 "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
  echo "WARNING: Ollama is not reachable at ${OLLAMA_URL} — the summary phase will be skipped." >&2
  echo "         Start it with:  ollama serve   (and: ollama pull qwen3:32b)" >&2
fi

export PYTHONPATH="${RAG_ROOT}:${JFR_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$RAG_ROOT"

echo "Building library — logging to $LOG"
echo "Leave this running. To background it:  nohup ./build_library.sh > /dev/null 2>&1 &"
echo

# Tee to a log so you can walk away and check progress later with: tail -f build_library.log
"$VENV/bin/python" build_library.py "$@" 2>&1 | tee -a "$LOG"
