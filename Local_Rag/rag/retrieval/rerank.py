"""
Reranker using BAAI/bge-reranker-v2-m3 via sentence-transformers.
Loaded once at server startup.
"""
import sys
from pathlib import Path
from typing import Sequence

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RERANKER_MODEL, RERANKER_CACHE

def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_DEVICE = _detect_device()


def load_reranker():
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(
        RERANKER_MODEL,
        max_length=512,
        device=_DEVICE,
        cache_folder=str(RERANKER_CACHE),
    )
    return model


def rerank(model, query: str, chunks: list[dict], top_n: int) -> list[dict]:
    """
    Rerank chunks using the cross-encoder. Returns top_n chunks sorted by score.
    Each chunk dict must have a 'text' field.
    """
    if not chunks:
        return []

    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs, show_progress_bar=False)

    scored = sorted(zip(scores, chunks), key=lambda x: -x[0])
    return [c for _, c in scored[:top_n]]
