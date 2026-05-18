import os
import streamlit as st
from loguru import logger

LOCAL_MODEL_PATH = "./models/embedding_model"


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    logger.info("Loading embedding model...")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        path = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "sentence-transformers/all-MiniLM-L6-v2"
        model = HuggingFaceEmbeddings(
            model_name=path,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info("Embedding model ready")
        return model
    except Exception as e:
        logger.error(f"Embedding model error: {e}")
        raise