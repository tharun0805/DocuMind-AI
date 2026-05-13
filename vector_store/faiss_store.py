import os
from langchain_community.vectorstores import FAISS
from embeddings.embedding_model import get_embedding_model
from loguru import logger


DB_PATH = "vector_db"


def create_vector_store(chunks: list[str]) -> FAISS:
    logger.info("Creating FAISS vector store...")

    embedding_model = get_embedding_model()
    vector_store = FAISS.from_texts(chunks, embedding_model)

    os.makedirs(DB_PATH, exist_ok=True)
    vector_store.save_local(DB_PATH)

    logger.info(f"Vector store created and saved to {DB_PATH}")
    return vector_store


def load_vector_store() -> FAISS:
    logger.info("Loading FAISS vector store from disk...")

    embedding_model = get_embedding_model()
    vector_store = FAISS.load_local(
        DB_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    logger.info("Vector store loaded successfully")
    return vector_store


def vector_search(query: str, k: int = 5) -> list[str]:
    vector_store = load_vector_store()
    results = vector_store.similarity_search(query, k=k)

    logger.info(f"Vector search returned {len(results)} results")
    return [doc.page_content for doc in results]