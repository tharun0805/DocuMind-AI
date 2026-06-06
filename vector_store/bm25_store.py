"""
vector_store/bm25_store.py

SPEED FIX:
- Keep built BM25Okapi in a module-level variable (_bm25_cache) so searches
  never need to hit disk.
- Write the pickle file in a background daemon thread so it never blocks the
  upload progress bar.
"""
import os
import pickle
import threading

import streamlit as st
from loguru import logger
from rank_bm25 import BM25Okapi

BM25_PATH = "vector_db/bm25_index.pkl"

# Module-level in-memory cache — set by create_bm25_index, read by bm25_search
_bm25_cache = None   # tuple: (BM25Okapi, chunks)


def create_bm25_index(chunks: list[str]) -> BM25Okapi:
    """Build BM25 index in RAM. Persist to disk in background (non-blocking)."""
    global _bm25_cache
    logger.debug(f"Building BM25 ({len(chunks)} chunks)...")
    tokenized   = [chunk.lower().split() for chunk in chunks]
    bm25        = BM25Okapi(tokenized)
    _bm25_cache = (bm25, chunks)   # immediately available for search

    # Write pickle to disk without blocking the caller
    def _save():
        try:
            os.makedirs("vector_db", exist_ok=True)
            with open(BM25_PATH, "wb") as fh:
                pickle.dump(_bm25_cache, fh)
        except Exception as exc:
            logger.warning(f"BM25 pickle save failed (non-critical): {exc}")

    threading.Thread(target=_save, daemon=True).start()

    # Clear Streamlit cache so get_cached_bm25 reloads on next cold start
    get_cached_bm25.clear()
    logger.debug("BM25 built (in-memory, disk save in background)")
    return bm25


@st.cache_resource(show_spinner=False)
def get_cached_bm25():
    """Cold-start loader — only called when _bm25_cache is empty (app restart)."""
    with open(BM25_PATH, "rb") as fh:
        return pickle.load(fh)


def load_bm25_index() -> tuple:
    return get_cached_bm25()


def bm25_search(query: str, k: int = 12) -> list[str]:
    # Prefer the fast in-memory cache; fall back to disk on cold restart
    data        = _bm25_cache if _bm25_cache is not None else get_cached_bm25()
    bm25, chunks = data
    scores      = bm25.get_scores(query.lower().split())
    top_idx     = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_idx     = [i for i in top_idx if scores[i] > 0][:k]
    return [chunks[i] for i in top_idx]
