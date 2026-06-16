#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# LabUI — unified launcher for the FastAPI server that serves both the
# Journal-Fit Recommender (submission_strategy) and the Local Papers RAG.
#
# Binds 127.0.0.1:8770 by default (localhost only). Set LABUI_HOST=0.0.0.0 to
# expose on your LAN, and LABUI_PORT to change the port.
# ─────────────────────────────────────────────────────────────────────────────
set -e

APPS_ROOT="$(cd "$(dirname "$0")" && pwd)"
JFR_ROOT="$APPS_ROOT/submission_strategy"
RAG_ROOT="$APPS_ROOT/Local_Rag/rag"
VENV="$JFR_ROOT/.venv"

# Self-contained instance: bind localhost only by default (opt into LAN with
# LABUI_HOST=0.0.0.0), with its own data tree under this folder.
HOST="${LABUI_HOST:-${CODEX_HOST:-127.0.0.1}}"
PORT="${LABUI_PORT:-${CODEX_PORT:-8770}}"
# Generator model is auto-detected from whatever Ollama has installed. Pin one
# only if you want to: export RAG_GEN_MODEL=llama3.1:8b (or LABUI_GEN_MODEL).
GEN_MODEL="${RAG_GEN_MODEL:-${LABUI_GEN_MODEL:-${CODEX_GEN_MODEL:-}}}"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
export RAG_GEN_MODEL="$GEN_MODEL"   # the app resolves "" → first installed model

# Keep all JFR data inside this LabUI folder.
export JFR_DATA_DIR="${JFR_DATA_DIR:-$APPS_ROOT/.data/jfr}"
export JFR_WEB_HOST="$HOST"
export JFR_WEB_PORT="$PORT"

# ── 1. Ollama reachable? ─────────────────────────────────────────────────────
if ! curl -fs --max-time 3 "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
  echo "ERROR: Ollama is not reachable at ${OLLAMA_URL}"
  echo "       Start it with:  ollama serve"
  exit 1
fi

# ── 2. A generator model is available? ───────────────────────────────────────
TAGS_JSON="$(curl -fs --max-time 3 "${OLLAMA_URL}/api/tags" || echo '{}')"
if [ -n "$GEN_MODEL" ]; then
  # A specific model was pinned — pull it if missing.
  if ! printf '%s' "$TAGS_JSON" | grep -q "\"${GEN_MODEL}\""; then
    echo "Pulling pinned model ${GEN_MODEL} (one-time download)…"
    ollama pull "${GEN_MODEL}"
  fi
elif ! printf '%s' "$TAGS_JSON" | grep -q '"name"'; then
  # Auto mode but Ollama has no models at all — guide, don't guess.
  echo "ERROR: Ollama has no models installed."
  echo "       Pull one first, e.g.:  ollama pull llama3.1:8b"
  echo "       (or any model you prefer — the app auto-detects it)."
  exit 1
else
  FIRST_MODEL="$(printf '%s' "$TAGS_JSON" | grep -o '"name":"[^"]*"' | head -1 | sed 's/.*:"//;s/"//')"
  echo "→ No model pinned; auto-detecting from Ollama (e.g. ${FIRST_MODEL})."
fi

# ── 3. Venv present? ─────────────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
  echo "ERROR: venv not found at $VENV"
  echo "       Create it with:"
  echo "         cd $JFR_ROOT && python3.12 -m venv .venv"
  echo "         source .venv/bin/activate && pip install -e ."
  exit 1
fi

# ── 4. Already-running guard (cross-platform: ss on Linux, lsof on macOS) ─────
port_in_use() {
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:\.]${PORT}\$"
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1
  else
    return 1   # can't tell — assume free
  fi
}
if port_in_use; then
  echo "WARNING: something is already listening on port ${PORT}."
  echo "         Stop it first, or set LABUI_PORT=<other> and retry."
  exit 1
fi

# ── 5. Compose PYTHONPATH ────────────────────────────────────────────────────
export PYTHONPATH="${JFR_ROOT}:${RAG_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

cd "$JFR_ROOT"

# ── 6. Boot ──────────────────────────────────────────────────────────────────
# LAN IP: hostname -I on Linux; ipconfig getifaddr on macOS.
if hostname -I >/dev/null 2>&1; then
  LAN_IP="$(hostname -I | awk '{print $1}')"
elif command -v ipconfig >/dev/null 2>&1; then
  LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null)"
else
  LAN_IP="$(hostname 2>/dev/null)"
fi
cat <<EOF

════════════════════════════════════════════════════════════════
  LabUI — Lab · Literature · Publication
  Your scientific record, end to end.

  Generator:  ${GEN_MODEL:-auto-detected}  (via Ollama)
  Local:      http://localhost:${PORT}
  LAN:        http://${LAN_IP:-<your-ip>}:${PORT}
  API docs:   http://localhost:${PORT}/api/docs
  Press Ctrl+C to stop.
════════════════════════════════════════════════════════════════

EOF

exec "$VENV/bin/python" -m uvicorn jfr.web.app:app \
  --host "$HOST" --port "$PORT" --no-access-log
