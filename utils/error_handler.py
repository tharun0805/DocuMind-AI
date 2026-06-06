from loguru import logger
 
 
# ── Custom Exception Hierarchy ─────────────────────────────────────
 
class DocuMindError(Exception):
    """Base exception for all DocuMind AI errors."""
    pass
 
 
class DocumentProcessingError(DocuMindError):
    """Raised when document ingestion or parsing fails."""
    pass
 
 
class EmbeddingError(DocuMindError):
    """Raised when embedding model fails."""
    pass
 
 
class RetrievalError(DocuMindError):
    """Raised when FAISS or BM25 retrieval fails."""
    pass
 
 
class AgentError(DocuMindError):
    """Raised when an AI agent fails to complete its task."""
    pass
 
 
class APIKeyError(DocuMindError):
    """Raised when an API key is missing or invalid."""
    pass
 
 
class RateLimitError(DocuMindError):
    """Raised when an API rate limit is hit."""
    pass
 
 
# ── User-Friendly Message Map ──────────────────────────────────────
 
_USER_MESSAGES = {
    "rate_limit": (
        "The AI service is currently busy (rate limit reached). "
        "Please wait 30-60 seconds and try again."
    ),
    "api_key": (
        "API key error. Please check your .env file and ensure "
        "GOOGLE_API_KEY and GROQ_API_KEY are correctly set."
    ),
    "file_not_found": (
        "The document file could not be found. "
        "Please re-upload your document and try again."
    ),
    "file_corrupt": (
        "The document appears to be corrupt or in an unsupported format. "
        "Please try a different file."
    ),
    "file_empty": (
        "The uploaded file is empty. "
        "Please upload a file with content."
    ),
    "embedding": (
        "The embedding model encountered an error. "
        "Please restart the app and try again."
    ),
    "retrieval": (
        "Could not retrieve relevant content from the document. "
        "Please try rephrasing your question."
    ),
    "network": (
        "A network error occurred. "
        "Please check your internet connection and try again."
    ),
    "timeout": (
        "The request timed out. "
        "The document may be too large. Please try a shorter question."
    ),
    "memory": (
        "Memory error — the document may be too large to process. "
        "Try uploading a smaller file."
    ),
    "generic": (
        "An unexpected error occurred. "
        "Please try again or restart the application."
    ),
}
 
 
# ── Error Classifier ───────────────────────────────────────────────
 
def _classify(error_msg: str, error_type: str) -> str:
    msg = error_msg.lower()
 
    if any(k in msg for k in ["429", "rate_limit", "resource_exhausted", "quota"]):
        return "rate_limit"
    if any(k in msg for k in ["api_key", "apikey", "invalid_api_key", "authentication"]):
        return "api_key"
    if "filenotfounderror" in error_type.lower() or "no such file" in msg:
        return "file_not_found"
    if any(k in msg for k in ["corrupt", "cannot read", "bad zip", "invalid pdf"]):
        return "file_corrupt"
    if "empty" in msg and "file" in msg:
        return "file_empty"
    if "embedding" in msg or "sentence" in msg:
        return "embedding"
    if any(k in msg for k in ["retrieval", "faiss", "bm25", "index"]):
        return "retrieval"
    if any(k in msg for k in ["connectionerror", "connection refused", "network"]):
        return "network"
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "memoryerror" in error_type.lower() or "out of memory" in msg:
        return "memory"
    return "generic"
 
 
# ── Public API ─────────────────────────────────────────────────────
 
def handle_error(error: Exception, context: str = "") -> str:
    """
    Log the full error internally and return a user-friendly message.
 
    Args:
        error:   The exception that was caught.
        context: A short label for where the error occurred.
                 e.g. "document_processing", "qa_agent", "retrieval"
 
    Returns:
        A safe, non-technical string suitable for display in the UI.
    """
    error_type = type(error).__name__
    error_msg = str(error)
    prefix = f"[{context}] " if context else ""
 
    # Full error logged internally
    logger.error(f"{prefix}{error_type}: {error_msg}")
    logger.exception(error)
 
    category = _classify(error_msg, error_type)
    return _USER_MESSAGES[category]
 
 
def safe_execute(func, *args, context: str = "", fallback=None, **kwargs):
    """
    Execute a function safely — returns fallback value on any exception.
 
    Usage:
        result = safe_execute(my_function, arg1, arg2, context="my_step")
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        handle_error(e, context=context)
        return fallback
 