"""Backwards-compatible entry-point.

The actual FastAPI app lives in :mod:`src.api.main` — import from there
in new code.  This module only re-exports ``app`` so ``uvicorn
src.main:app`` and existing Docker configs continue to work.
"""

from src.api.main import app

__all__ = ["app"]
