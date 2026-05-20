import os
import streamlit as st
from loguru import logger

LOCAL_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "embedding_model"
)


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings

    if os.path.exists(LOCAL_MODEL_PATH):
        logger.info("Loading embedding model from local cache")
        path = LOCAL_MODEL_PATH
    else:
        logger.info("Local model not found — downloading from HuggingFace")
        path = "sentence-transformers/all-MiniLM-L6-v2"

    model = HuggingFaceEmbeddings(
        model_name=path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    logger.info("Embedding model ready")
    return model