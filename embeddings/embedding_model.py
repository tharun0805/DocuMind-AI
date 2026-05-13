from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger


def get_embedding_model() -> HuggingFaceEmbeddings:
    logger.info("Loading embedding model...")

    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    logger.info("Embedding model loaded successfully")
    return model