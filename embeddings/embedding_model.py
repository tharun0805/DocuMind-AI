import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger


@st.cache_resource(show_spinner=False)
def get_embedding_model() -> HuggingFaceEmbeddings:
    logger.info("Loading embedding model...")
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    logger.info("Embedding model loaded and cached")
    return model