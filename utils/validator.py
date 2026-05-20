import os
from loguru import logger

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".ppt",
    ".xlsx", ".xls", ".csv", ".txt", ".md"
}


def validate_file(file_path: str) -> tuple[bool, str]:
    if not os.path.exists(file_path):
        return False, "File not found. Please upload again."

    extension = os.path.splitext(file_path)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file type: {extension}. Supported: PDF, DOCX, PPTX, XLSX, CSV, TXT"

    if os.path.getsize(file_path) == 0:
        return False, "File is empty."

    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    logger.info(f"File validated: {file_path} ({size_mb:.1f}MB)")
    return True, "Valid"


def validate_question(question: str) -> tuple[bool, str]:
    if not question or not question.strip():
        return False, "Question cannot be empty."
    if len(question.strip()) < 2:
        return False, "Question too short."
    return True, "Valid"
