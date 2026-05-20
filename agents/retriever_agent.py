from retrieval.hybrid_retriever import hybrid_search
from loguru import logger


def retrieve_context(question: str, k: int = 6) -> str:
    logger.info(f"Retrieving context for: {question}")

    chunks = hybrid_search(question, k=k)

    if not chunks:
        logger.warning("No chunks found - using broad fallback")
        try:
            from vector_store.faiss_store import load_vector_store

            vs = load_vector_store()
            results = vs.similarity_search(question, k=8)
            chunks = [doc.page_content for doc in results]
        except Exception as e:
            logger.error(f"Fallback failed: {e}")
            chunks = []

    if not chunks:
        return "No relevant content found in the document."

    context = "\n\n".join(chunks)
    logger.info(f"Retrieved {len(chunks)} chunks")
    return context
