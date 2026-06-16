#!/usr/bin/env python3
"""
Graph extraction pipeline: uses GLiNER and spaCy to pull entities and
relations from each chunk, then stores them in graph.db.

Run standalone:
    python -m ingest.extract_graph          # extract all un-processed chunks
    python -m ingest.extract_graph --reset  # wipe graph.db and re-extract
"""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

# allow running as __main__
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DB_PATH
from retrieval.graph import (
    _open_db as _open_graph_db,
    GRAPH_DB_PATH,
    init_graph_db,
    upsert_entity,
    upsert_relation,
    link_entity_chunk,
    set_graph_meta,
)

try:
    from gliner import GLiNER
except ImportError:
    print("Error: gliner is not installed. Please run: pip install gliner")
    sys.exit(1)

try:
    import spacy
except ImportError:
    print("Error: spacy is not installed. Please run: pip install spacy")
    sys.exit(1)

_VALID_TYPES = ["Chemical", "Method", "Concept", "Measurement", "Material", "Other"]

# Load models lazily
_gliner_model = None
_spacy_model = None

def load_models():
    global _gliner_model, _spacy_model
    if _gliner_model is None:
        print("[graph] Loading GLiNER model (urchade/gliner_medium-v2.1)...")
        _gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    if _spacy_model is None:
        print("[graph] Loading spaCy model (en_core_web_sm)...")
        try:
            _spacy_model = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading en_core_web_sm...")
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            _spacy_model = spacy.load("en_core_web_sm")


def _extract_entities_and_relations(text: str) -> dict:
    """Extract entities using GLiNER and relations using spaCy heuristics."""
    if not text.strip():
        return {"entities": [], "relations": []}
        
    doc = _spacy_model(text)
    
    # 1. GLiNER Entities
    entities = []
    # Predict entities sentence by sentence
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        preds = _gliner_model.predict_entities(sent_text, _VALID_TYPES, threshold=0.5)
        for p in preds:
            name = p["text"].strip()
            # Basic filtering
            if len(name) < 2 or name.lower() in ["the", "a", "an", "this", "that", "it"]:
                continue
            entities.append({
                "name": name,
                "type": p["label"],
                "description": "", # GLiNER doesn't provide descriptions
                "sent_idx": sent.start 
            })
            
    # Deduplicate entities by name
    unique_entities = {}
    for e in entities:
        name_lower = e["name"].lower()
        if name_lower not in unique_entities:
            unique_entities[name_lower] = e
            
    final_entities = list(unique_entities.values())
    
    # 2. Heuristic Relations
    relations = []
    for sent in doc.sents:
        sent_entities = [e for e in entities if e["sent_idx"] == sent.start]
        if len(sent_entities) >= 2:
            for i in range(len(sent_entities)):
                for j in range(i + 1, len(sent_entities)):
                    e1 = sent_entities[i]
                    e2 = sent_entities[j]
                    
                    verbs = [token.lemma_ for token in sent if token.pos_ == "VERB"]
                    if verbs:
                        relation = verbs[0]
                        relations.append({
                            "source": e1["name"],
                            "target": e2["name"],
                            "relation": relation,
                            "description": f"Co-occur in sentence with verb '{relation}'"
                        })
                        
    return {
        "entities": final_entities,
        "relations": relations
    }


def _open_rag_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def run_extraction(
    batch_size: int = 50,
    skip_processed: bool = True,
    progress_cb=None,
) -> dict:
    load_models()
    init_graph_db()
    rag = _open_rag_db()
    graph = _open_graph_db()

    processed: set[str] = set()
    if skip_processed:
        rows = graph.execute("SELECT DISTINCT chunk_id FROM entity_chunks").fetchall()
        processed = {r["chunk_id"] for r in rows}

    all_chunks = rag.execute(
        "SELECT chunk_id, paper_id, text, section_name FROM chunks"
    ).fetchall()
    rag.close()

    to_process = [c for c in all_chunks if c["chunk_id"] not in processed]
    total = len(to_process)
    print(f"[graph] {total} chunks to process ({len(processed)} already done)")

    stats = {"processed": 0, "entities": 0, "relations": 0, "errors": 0}
    start = time.time()

    for i, chunk in enumerate(to_process):
        chunk_id = chunk["chunk_id"]
        paper_id = chunk["paper_id"]
        text = chunk["text"] or ""

        if len(text.strip()) < 80:
            continue

        extracted = _extract_entities_and_relations(text)

        name_to_id: dict[str, str] = {}

        for ent in extracted.get("entities", []):
            name = (ent.get("name") or "").strip()
            etype = ent.get("type", "Other")
            desc = (ent.get("description") or "").strip()
            if not name or len(name) < 2:
                continue
            if etype not in _VALID_TYPES:
                etype = "Other"
            try:
                eid = upsert_entity(graph, name, etype, desc, paper_id)
                link_entity_chunk(graph, eid, chunk_id, paper_id)
                name_to_id[name.lower()] = eid
                stats["entities"] += 1
            except Exception as e:
                print(f"  [graph] entity upsert error: {e}")
                stats["errors"] += 1

        for rel in extracted.get("relations", []):
            src_name = (rel.get("source") or "").strip().lower()
            tgt_name = (rel.get("target") or "").strip().lower()
            relation = (rel.get("relation") or "").strip()
            desc = (rel.get("description") or "").strip()
            if not src_name or not tgt_name or not relation:
                continue
            
            src_id = name_to_id.get(src_name)
            tgt_id = name_to_id.get(tgt_name)
            if not src_id:
                row = graph.execute(
                    "SELECT entity_id FROM entities WHERE name_lower=?", (src_name,)
                ).fetchone()
                if row:
                    src_id = row["entity_id"]
            if not tgt_id:
                row = graph.execute(
                    "SELECT entity_id FROM entities WHERE name_lower=?", (tgt_name,)
                ).fetchone()
                if row:
                    tgt_id = row["entity_id"]
            if src_id and tgt_id:
                try:
                    upsert_relation(graph, src_id, tgt_id, relation, desc, paper_id)
                    stats["relations"] += 1
                except Exception as e:
                    print(f"  [graph] relation upsert error: {e}")

        graph.commit()
        stats["processed"] += 1

        if progress_cb:
            progress_cb(i + 1, total, stats)
        elif (i + 1) % 10 == 0 or i == total - 1:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"  [graph] {i+1}/{total} chunks | "
                f"{stats['entities']} entities | {stats['relations']} relations | "
                f"ETA {eta:.0f}s"
            )

    set_graph_meta("last_extraction_at", str(time.time()))
    set_graph_meta("papers_indexed", str(
        graph.execute("SELECT COUNT(DISTINCT paper_id) FROM entity_chunks").fetchone()[0]
    ))
    graph.commit()
    graph.close()

    elapsed = time.time() - start
    print(
        f"[graph] Done in {elapsed:.1f}s — "
        f"{stats['entities']} entities, {stats['relations']} relations "
        f"from {stats['processed']} chunks"
    )
    return stats


def reset_graph_db() -> None:
    if GRAPH_DB_PATH.exists():
        GRAPH_DB_PATH.unlink()
        print("[graph] graph.db deleted")
    init_graph_db()
    print("[graph] graph.db re-initialised")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract entities/relations from chunks into graph.db")
    parser.add_argument("--reset", action="store_true", help="Wipe graph.db before extraction")
    args = parser.parse_args()

    if args.reset:
        reset_graph_db()

    run_extraction()
