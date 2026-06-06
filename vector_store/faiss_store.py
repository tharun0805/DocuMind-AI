"""
vector_store/faiss_store.py

SPEED FIX — two-phase FAISS:
- _faiss_cache holds the live FAISS instance (None until built).
- create_vector_store_background() builds FAISS in a daemon thread and
  swaps _faiss_cache the moment it finishes, without blocking the caller.
- vector_search() falls back gracefully to an empty list while FAISS is
  still building — hybrid_retriever then relies solely on BM25 for those
  first few seconds, which is perfectly fine.
- Disk save is also backgrounded (unchanged from previous patch).
"""
import os
import threading

import streamlit as st
from langchain_community.vectorstores import FAISS
from loguru import logger

from embeddings.embedding_model import get_embedding_model

DB_PATH = "vector_db"

# Module-level in-memory cache — None while FAISS is still being built
_faiss_cache = None          # FAISS | None
_faiss_lock  = threading.Lock()


def faiss_enabled() -> bool:
    return True


def faiss_ready() -> bool:
    """True once the background build has completed."""
    return _faiss_cache is not None


def create_vector_store(chunks: list[str]) -> FAISS:
    """
    Synchronous build — used by switch_document().
    For uploads, use create_vector_store_background() instead.
    """
    global _faiss_cache
    logger.debug(f"Building FAISS synchronously ({len(chunks)} chunks)...")
    embedding_model = get_embedding_model()
    vs = FAISS.from_texts(chunks, embedding_model)
    with _faiss_lock:
        _faiss_cache = vs

    def _save():
        try:
            os.makedirs(DB_PATH, exist_ok=True)
            vs.save_local(DB_PATH)
        except Exception as exc:
            logger.warning(f"FAISS save_local failed (non-critical): {exc}")

    threading.Thread(target=_save, daemon=True).start()
    get_cached_vector_store.clear()
    logger.debug("FAISS built (sync)")
    return vs


def create_vector_store_background(chunks: list[str]) -> None:
    """
    Non-blocking build: runs in a daemon thread, swaps _faiss_cache when done.
    Call this AFTER setting doc_ready=True so the UI unlocks immediately.
    The retriever uses BM25-only until FAISS finishes (~10-20s later).
    """
    global _faiss_cache
    # Clear stale cache from previous document
    with _faiss_lock:
        _faiss_cache = None

    def _build():
        global _faiss_cache
        try:
            logger.debug(f"FAISS background build ({len(chunks)} chunks)...")
            embedding_model = get_embedding_model()
            vs = FAISS.from_texts(chunks, embedding_model)
            with _faiss_lock:
                _faiss_cache = vs
            logger.debug("FAISS background build complete — now live")

            def _save():
                try:
                    os.makedirs(DB_PATH, exist_ok=True)
                    vs.save_local(DB_PATH)
                except Exception as exc:
                    logger.warning(f"FAISS save_local failed (non-critical): {exc}")

            threading.Thread(target=_save, daemon=True).start()
            get_cached_vector_store.clear()
        except Exception as e:
            logger.error(f"FAISS background build failed: {e}")

    threading.Thread(target=_build, daemon=True).start()


@st.cache_resource(show_spinner=False)
def get_cached_vector_store() -> FAISS:
    """Cold-start loader — only called on app restart."""
    embedding_model = get_embedding_model()
    return FAISS.load_local(
        DB_PATH,
        embedding_model,
        allow_dangerous_deserialization=True,
    )


def load_vector_store() -> FAISS:
    return get_cached_vector_store()


def vector_search(query: str, k: int = 12) -> list[str]:
    """
    Returns results if FAISS is ready; empty list if still building.
    hybrid_retriever falls back to BM25-only during the build window.
    """
    with _faiss_lock:
        vs = _faiss_cache
    if vs is None:
        logger.debug("FAISS not ready yet — BM25-only for this query")
        return []
    results = vs.similarity_search(query, k=k)
    return [doc.page_content for doc in results]
