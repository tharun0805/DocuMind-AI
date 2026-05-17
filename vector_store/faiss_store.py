import os
import streamlit as st
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
    logger.info(f"Vector store saved to {DB_PATH}")
    get_cached_vector_store.clear()
    return vector_store


@st.cache_resource(show_spinner=False)
def get_cached_vector_store() -> FAISS:
    logger.info("Loading FAISS from cache...")
    embedding_model = get_embedding_model()
    vector_store = FAISS.load_local(
        DB_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return vector_store


def load_vector_store() -> FAISS:
    return get_cached_vector_store()


def vector_search(query: str, k: int = 6) -> list[str]:
    vector_store = load_vector_store()
    results = vector_store.similarity_search(query, k=k)
    logger.info(f"Vector search returned {len(results)} results")
    return [doc.page_content for doc in results]