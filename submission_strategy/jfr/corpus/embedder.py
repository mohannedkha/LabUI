"""Local embedding pipeline using sentence-transformers (specter2 + bge-large).

All Qdrant-backed searches use specter2 (768-dim) so dimensions are consistent
across corpus and query vectors.  bge-large is loaded lazily for re-ranking only.
"""

from __future__ import annotations

import sqlite3

import numpy as np

# Cache keyed by model_name so multiple models coexist in the same process.
_model_cache: dict[str, object] = {}


def _load_model(model_name: str):
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_texts(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    """Embed texts with the named model; returns float32 array (N, dim)."""
    model = _load_model(model_name)
    return model.encode(  # type: ignore[return-value]
        texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True
    )


def embed_corpus(
    conn: sqlite3.Connection,
    journal_id: str,
    qdrant_client,
    collection_name: str,
    model_name: str,
    batch_size: int = 64,
    progress_cb=None,
) -> int:
    """Embed all un-vectorised corpus articles for a journal and upsert into Qdrant."""
    from qdrant_client.models import PointStruct

    rows = conn.execute(
        """SELECT id, doi, title, abstract FROM corpus_article
           WHERE journal_id=? AND (vector_id IS NULL OR embedding_model != ?)
           AND abstract IS NOT NULL AND abstract != ''""",
        (journal_id, model_name),
    ).fetchall()

    if not rows:
        return 0

    ids = [r["id"] for r in rows]
    texts = [f"{r['title']}. {r['abstract']}" for r in rows]

    embedded = 0
    for i in range(0, len(texts), batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]
        vecs = embed_texts(batch_texts, model_name, batch_size=batch_size)
        points = [
            PointStruct(id=bid, vector=vec.tolist(), payload={"journal_id": journal_id, "doi": rows[i + j]["doi"]})
            for j, (bid, vec) in enumerate(zip(batch_ids, vecs))
        ]
        qdrant_client.upsert(collection_name=collection_name, points=points)
        for bid in batch_ids:
            conn.execute(
                "UPDATE corpus_article SET vector_id=?, embedding_model=? WHERE id=?",
                (str(bid), model_name, bid),
            )
        conn.commit()
        embedded += len(batch_ids)
        if progress_cb:
            progress_cb(embedded, len(ids))

    return embedded


def embed_manuscript_fields(
    abstract: str,
    principal_claim: str,
    abstract_model: str,
    claim_model: str,
) -> dict[str, np.ndarray]:
    """Return embedding vectors for manuscript abstract and principal claim."""
    abs_vec = embed_texts([abstract], abstract_model)[0]
    claim_vec = embed_texts([principal_claim], claim_model)[0]
    return {"abstract": abs_vec, "claim": claim_vec}
