"""
Embedding helpers for SecondSelf.

Wraps ``sentence-transformers`` with the ``all-MiniLM-L6-v2`` model
(384-dim vectors) for semantic similarity and linking.

All embeddings are persisted in ``data/embeddings.pkl``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .storage import DATA_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.pkl"

_model = None  # cached SentenceTransformer instance


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """
    Load and cache the ``all-MiniLM-L6-v2`` sentence-transformers model.

    The model is downloaded on first use (~80 MB) and cached locally
    by the ``sentence-transformers`` library.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_text(text: str) -> np.ndarray:
    """
    Embed *text* into a 384-dimensional vector.

    Parameters
    ----------
    text : str
        The text to embed.

    Returns
    -------
    np.ndarray
        A 384-dim float32 vector.
    """
    model = load_model()
    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text)
    vec = model.encode(text, convert_to_numpy=True)
    return vec.astype(np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed multiple texts in a single batch (more efficient).

    Parameters
    ----------
    texts : list[str]
        List of texts to embed.

    Returns
    -------
    np.ndarray
        Array of shape (n, 384).
    """
    model = load_model()
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Parameters
    ----------
    a, b : np.ndarray
        Vectors to compare.

    Returns
    -------
    float
        Similarity score in [-1, 1].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between *query* and each row of *matrix*.

    Parameters
    ----------
    query : np.ndarray
        Single vector (384,).
    matrix : np.ndarray
        Array of shape (n, 384).

    Returns
    -------
    np.ndarray
        Array of shape (n,) with similarity scores.
    """
    if matrix.shape[0] == 0:
        return np.array([])
    # Normalize
    query_norm = query / (np.linalg.norm(query) + 1e-8)
    matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return matrix_norms @ query_norm


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_embeddings() -> Dict[str, np.ndarray]:
    """
    Load embeddings from ``data/embeddings.pkl``.

    Returns
    -------
    dict
        ``{note_id: np.ndarray}`` — empty dict if file doesn't exist.
    """
    if not EMBEDDINGS_FILE.exists():
        return {}
    with open(EMBEDDINGS_FILE, "rb") as f:
        return pickle.load(f)


def save_embeddings(embeddings: Dict[str, np.ndarray]) -> None:
    """
    Save embeddings to ``data/embeddings.pkl``.

    Parameters
    ----------
    embeddings : dict
        ``{note_id: np.ndarray}``
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)