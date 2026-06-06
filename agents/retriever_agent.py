from loguru import logger


def retrieve_context(question: str, k: int = 8) -> str:
    """Increased k from 6→8 for richer context on book-type documents."""
    from retrieval.hybrid_retriever import hybrid_search
    try:
        chunks  = hybrid_search(question, k=k)
        context = "\n\n---\n\n".join(chunks) if chunks else ""
        logger.debug(f"Retrieved {len(chunks)} chunks")
        return context
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return ""