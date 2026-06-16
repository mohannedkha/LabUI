"""
BGE-M3 embedding module (sentence-transformers).

Loaded once at server startup; also used by build_index and memory.

Replaces the former SPECTER2 + proximity-adapter encoder. bge-m3 is a 1024-dim,
8192-token-context retriever, so our ~800-token chunks are embedded in full (the
old SPECTER2 path truncated at 512 tokens). Embeddings are L2-normalized, so the
sqlite-vec default L2 distance ranks identically to cosine similarity.

Public API is unchanged so existing call sites keep working:
  load_embed_model() -> (tokenizer_or_None, model)
  embed_texts(tok, model, texts, titles=None) -> np.ndarray (N, 1024)
  embed_query(tok, model, query) -> np.ndarray (1024,)
The `titles` argument is accepted for backward compatibility but ignored:
bge-m3 is symmetric and needs no title[SEP]body convention, and chunk text
already carries a "[title — section]" header.
"""
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBED_MODEL_BASE, EMBED_BATCH_SIZE


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_DEVICE = _detect_device()


def load_embed_model():
    """Return (None, SentenceTransformer) on the detected device.

    First element is None (no separate tokenizer needed) to preserve the
    (tokenizer, model) tuple shape that existing call sites unpack.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL_BASE, device=_DEVICE)
    # bge-m3 supports up to 8192 tokens; make sure ST doesn't cap lower.
    model.max_seq_length = 8192
    return None, model


def embed_texts(
    tokenizer,
    model,
    texts: Sequence[str],
    titles: Sequence[str] | None = None,
) -> np.ndarray:
    """Embed a batch of texts. Returns float32 array of shape (N, 1024)."""
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)
    vecs = model.encode(
        list(texts),
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vecs.astype(np.float32)


def embed_query(tokenizer, model, query: str) -> np.ndarray:
    """Embed a single query string. Returns shape (1024,)."""
    vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    return vec.astype(np.float32)
