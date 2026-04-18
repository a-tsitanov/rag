"""Sparse vector encoder for BM25-like keyword search in Milvus.

Uses feature hashing (hashing trick) to map tokens to sparse vector
dimensions.  No model fitting required — works immediately on any text.

Term weighting: sublinear TF ``1 + log(tf)`` per token.  True IDF is
omitted for simplicity; Milvus ``IP`` metric on sparse vectors combined
with ``RRFRanker`` provides good-enough keyword matching.
"""

from __future__ import annotations

import math
import re
from hashlib import md5

# 2^20 ≈ 1M dimensions — large enough to avoid hash collisions for
# vocabularies up to ~100K tokens (birthday paradox threshold).
MAX_DIM = 1 << 20

_TOKEN_RE = re.compile(r"[a-zA-Z0-9\u0400-\u04FF]{2,}")


def _tokenize(text: str) -> list[str]:
    """Lowercase alpha-numeric tokens, min length 2."""
    return _TOKEN_RE.findall(text.lower())


def _hash_token(token: str) -> int:
    """Deterministic hash → dimension index in [0, MAX_DIM)."""
    return int(md5(token.encode(), usedforsecurity=False).hexdigest()[:8], 16) % MAX_DIM


class SparseEncoder:
    """Stateless sparse encoder — no fitting, no saved state."""

    def encode_document(self, text: str) -> dict[int, float]:
        """Encode document text into a sparse vector (dim → weight).

        Weight = ``1 + log(tf)`` (sublinear term frequency).
        """
        tokens = _tokenize(text)
        if not tokens:
            return {0: 0.0}  # Milvus requires non-empty sparse vector

        tf: dict[int, int] = {}
        for tok in tokens:
            dim = _hash_token(tok)
            tf[dim] = tf.get(dim, 0) + 1

        return {dim: 1.0 + math.log(count) for dim, count in tf.items()}

    def encode_query(self, text: str) -> dict[int, float]:
        """Encode query text into a sparse vector.

        For queries, binary weighting (1.0 per unique token) works well
        — we want to match any keyword, not weight by repetition.
        """
        tokens = _tokenize(text)
        if not tokens:
            return {0: 0.0}

        dims = {_hash_token(tok) for tok in tokens}
        return {dim: 1.0 for dim in dims}
