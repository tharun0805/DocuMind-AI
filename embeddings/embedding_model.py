import os
import threading
from loguru import logger

os.environ["TRANSFORMERS_VERBOSITY"]        = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]        = "false"

_model      = None
_model_lock = threading.Lock()


def get_embedding_model():
    """
    Thread-safe singleton — replaces @st.cache_resource.
    Works from any background thread without ScriptRunContext warnings.
    This is the permanent fix for parallel FAISS indexing.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                logger.debug("Loading embedding model...")
                from langchain_huggingface import HuggingFaceEmbeddings
                _model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={
                        "normalize_embeddings": True,
                        "batch_size": 64,
                    },
                    cache_folder=".cache/embeddings",
                )
                logger.debug("Embedding model ready")
    return _model