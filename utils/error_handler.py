from loguru import logger


class DocuMindError(Exception):
    pass


class DocumentProcessingError(DocuMindError):
    pass


class RetrievalError(DocuMindError):
    pass


class AgentError(DocuMindError):
    pass


def handle_error(error: Exception, context: str = "") -> str:
    error_type = type(error).__name__
    error_msg = str(error)

    if context:
        logger.error(f"[{context}] {error_type}: {error_msg}")
    else:
        logger.error(f"{error_type}: {error_msg}")

    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
        return "The AI service is temporarily rate limited. Please wait a moment and try again."

    if "API_KEY" in error_msg or "api_key" in error_msg:
        return "API key error. Please check your .env file."

    if "FileNotFoundError" in error_type:
        return "Document file not found. Please upload the document again."

    if "PermissionError" in error_type:
        return "Permission denied reading the file. Please try again."

    if "FAISS" in error_msg or "vector" in error_msg.lower():
        return "Search index error. Please process the document again."

    return f"An unexpected error occurred: {error_msg}"