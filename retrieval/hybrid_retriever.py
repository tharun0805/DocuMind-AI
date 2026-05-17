from vector_store.faiss_store import vector_search
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

    sorted_chunks = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return [chunk for chunk, score in sorted_chunks]


def hybrid_search(query: str, k: int = 6) -> list[str]:
    logger.info(f"Running hybrid search for: {query}")

    try:
        vector_results = vector_search(query, k=k)
    except Exception as e:
        logger.warning(f"Vector search failed: {e}")
        vector_results = []

    try:
        bm25_results = bm25_search(query, k=k)
    except Exception as e:
        logger.warning(f"BM25 search failed: {e}")
        bm25_results = []

    if not vector_results and not bm25_results:
        logger.warning("Both searches returned empty results")
        return []

    fused_results = reciprocal_rank_fusion(vector_results, bm25_results)

    logger.info(f"Hybrid search returned {len(fused_results[:k])} results")
    return fused_results[:k]