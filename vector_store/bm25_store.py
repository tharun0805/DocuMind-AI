import pickle
import os
import streamlit as st
from rank_bm25 import BM25Okapi
from loguru import logger


BM25_PATH = "vector_db/bm25_index.pkl"


def create_bm25_index(chunks: list[str]) -> BM25Okapi:
    logger.info("Creating BM25 index...")
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    os.makedirs("vector_db", exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump((bm25, chunks), f)
    logger.info(f"BM25 index saved to {BM25_PATH}")
    get_cached_bm25.clear()
    return bm25


@st.cache_resource(show_spinner=False)
def get_cached_bm25() -> tuple:
    logger.info("Loading BM25 from cache...")
    with open(BM25_PATH, "rb") as f:
        bm25, chunks = pickle.load(f)
    return bm25, chunks


def load_bm25_index() -> tuple:
    return get_cached_bm25()


def bm25_search(query: str, k: int = 6) -> list[str]:
    bm25, chunks = load_bm25_index()
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]
    results = [chunks[i] for i in top_indices]
    logger.info(f"BM25 search returned {len(results)} results")
    return results