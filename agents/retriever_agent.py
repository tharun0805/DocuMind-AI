from retrieval.hybrid_retriever import hybrid_search
from vector_store.faiss_store import load_vector_store
from loguru import logger


def retrieve_context(question: str, k: int = 5) -> str:
    logger.info(f"Retrieving context for: {question}")

    chunks = hybrid_search(question, k=k)

    if not chunks or all(len(c.strip()) < 20 for c in chunks):
        logger.info("Hybrid search returned weak results. Falling back to broad retrieval...")
        try:
            vector_store = load_vector_store()
            broad_results = vector_store.similarity_search(question, k=8)
            chunks = [doc.page_content for doc in broad_results]
            logger.info(f"Broad retrieval returned {len(chunks)} chunks")
        except Exception as e:
            logger.warning(f"Broad retrieval failed: {e}")

    context = "\n\n".join(chunks)
    logger.info(f"Retrieved {len(chunks)} chunks as context")
    return context