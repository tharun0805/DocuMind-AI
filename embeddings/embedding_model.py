import os
import streamlit as st
from loguru import logger


LOCAL_MODEL_PATH = "./models/embedding_model"


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    logger.info("Loading embedding model...")

    if os.path.exists(LOCAL_MODEL_PATH):
        from langchain_huggingface import HuggingFaceEmbeddings
        model = HuggingFaceEmbeddings(
            model_name=LOCAL_MODEL_PATH,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info("Embedding model loaded from local cache")
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info("Embedding model loaded from HuggingFace")

    return model