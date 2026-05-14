import os
from loguru import logger


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".txt"}
MAX_FILE_SIZE_MB = 50


def validate_file(file_path: str) -> tuple[bool, str]:
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return False, "File not found. Please upload again."

    extension = os.path.splitext(file_path)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        logger.warning(f"Unsupported file type: {extension}")
        return False, f"Unsupported file type: {extension}. Supported: PDF, DOCX, PPTX, XLSX, CSV, TXT"

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        logger.warning(f"File too large: {file_size_mb:.1f}MB")
        return False, f"File too large: {file_size_mb:.1f}MB. Maximum allowed: {MAX_FILE_SIZE_MB}MB"

    if os.path.getsize(file_path) == 0:
        logger.warning(f"Empty file: {file_path}")
        return False, "File is empty. Please upload a valid document."

    logger.info(f"File validated: {file_path} ({file_size_mb:.1f}MB)")
    return True, "File is valid"


def validate_question(question: str) -> tuple[bool, str]:
    if not question or not question.strip():
        return False, "Question cannot be empty."

    if len(question.strip()) < 3:
        return False, "Question is too short. Please ask a complete question."

    if len(question) > 1000:
        return False, "Question is too long. Please keep it under 1000 characters."

    return True, "Question is valid"