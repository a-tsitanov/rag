"""Thin wrapper kept for backward compatibility.

The actual LightRAG singleton lives in ``src.retrieval.lightrag_setup``.
This module re-exports ``get_rag`` so existing code that imported from
``src.core.rag`` keeps working.
"""

from src.retrieval.lightrag_setup import get_rag

__all__ = ["get_rag"]
