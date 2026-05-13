from retrieval.hybrid_retriever import hybrid_search
from loguru import logger


def retrieve_context(question: str, k: int = 5) -> str:
    logger.info(f"Retrieving context for: {question}")

    chunks = hybrid_search(question, k=k)

    context = "\n\n".join(chunks)

    logger.info(f"Retrieved {len(chunks)} chunks as context")
    return context