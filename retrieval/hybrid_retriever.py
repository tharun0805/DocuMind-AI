from vector_store.bm25_store import bm25_search
from loguru import logger


def reciprocal_rank_fusion(
    vector_results: list[str],
    bm25_results: list[str],
    k: int = 60
) -> list[str]:
    scores = {}

    for rank, chunk in enumerate(vector_results):
        scores[chunk] = scores.get(chunk, 0) + 1 / (k + rank + 1)

    for rank, chunk in enumerate(bm25_results):
        scores[chunk] = scores.get(chunk, 0) + 1 / (k + rank + 1)

    return [c for c, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def hybrid_search(query: str, k: int = 12) -> list[str]:
    logger.debug(f"Hybrid search: {query[:50]}")

    vector_results = []
    try:
        from vector_store.faiss_store import faiss_enabled, vector_search

        if faiss_enabled():
            vector_results = vector_search(query, k=k)
    except Exception as e:
        logger.warning(f"Vector search failed: {e}")

    try:
        bm25_results = bm25_search(query, k=k)
    except Exception as e:
        logger.warning(f"BM25 search failed: {e}")
        bm25_results = []

    if not vector_results and not bm25_results:
        return []

    fused = reciprocal_rank_fusion(vector_results, bm25_results)

    logger.debug(f"Hybrid search: {len(fused[:k])} results")
    return fused[:k]
