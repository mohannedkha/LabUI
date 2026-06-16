#!/usr/bin/env bash
# Full library rebuild: parse -> embed(bge-m3) -> graph(GLiNER) -> memory migrate -> verify.
# Designed to run unattended in the background. Logs to logs/rebuild.log.
set -uo pipefail
cd "$(dirname "$0")"
source venv/bin/activate

TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/rebuild.log
mkdir -p logs
exec > >(tee -a "$LOG") 2>&1

say() { echo; echo "==== [$(date +%H:%M:%S)] $* ===="; }
fail() { echo "REBUILD_FAILED at stage: $*"; echo "FAILED" > data/REBUILD_STATUS; exit 1; }

echo "RUNNING" > data/REBUILD_STATUS
say "REBUILD START $TS"

# 0) Safety: back up memory.db (rag.db + graph.db already in dedup_backup_*)
cp -f data/memory.db "data/memory.db.pre_bgem3_$TS" 2>/dev/null || true

# 1) Move old index aside (kept, not deleted)
say "Archiving old rag.db + graph.db"
for f in rag.db graph.db; do
  for ext in "" "-wal" "-shm"; do
    [ -e "data/$f$ext" ] && mv "data/$f$ext" "data/$f.pre_bgem3_$TS$ext"
  done
done

# 2) Fresh parse: clear parsed JSON (backed up in dedup_backup_*) and re-parse PDFs
say "Wiping parsed JSON and re-parsing PDFs (Docling)"
rm -f data/parsed/*.json
python3 -m ingest.parse_pdfs || fail "parse_pdfs"

# 3) Build index with bge-m3 (+ basic DOI graph), content-hash dedup guard
say "Building index (bge-m3 embeddings)"
python3 -m ingest.build_index || fail "build_index"

# 4) Knowledge graph re-extraction (GLiNER + spaCy) — the long pole (~5h)
say "Extracting knowledge graph (GLiNER)"
python3 -m ingest.extract_graph || fail "extract_graph"

# 5) Migrate RAG memory vectors to 1024-dim
say "Migrating memory.db vectors to 1024"
python3 migrate_memory_1024.py || fail "memory_migrate"

# 6) Verify integrity
say "Verifying"
python3 verify_rebuild.py || fail "verify"

say "REBUILD COMPLETE $TS"
echo "DONE" > data/REBUILD_STATUS
