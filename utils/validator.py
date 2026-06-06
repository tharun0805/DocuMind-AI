import os
import re
from pathlib import Path
from loguru import logger
 
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc",
    ".pptx", ".ppt",
    ".xlsx", ".xls",
    ".csv", ".txt", ".md"
}
 
# Prompt injection patterns to block
_INJECTION_PATTERNS = [
    r"ignore (previous|above|all) instructions",
    r"you are now",
    r"act as (a|an)",
    r"forget (everything|all)",
    r"system prompt",
    r"jailbreak",
    r"do anything now",
]
 
 
def validate_file(file_path: str) -> tuple[bool, str]:
    """
    Validate an uploaded file.
 
    Returns:
        (True, "Valid") on success
        (False, "reason") on failure
    """
    path = Path(file_path)
 
    # Existence check
    if not path.exists():
        logger.warning(f"[VALIDATOR] File not found: {file_path}")
        return False, "File not found. Please upload again."
 
    # Extension check
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        logger.warning(f"[VALIDATOR] Unsupported extension: {ext}")
        return False, (
            f"Unsupported file type: '{ext}'. "
            f"Supported formats: PDF, DOCX, PPTX, XLSX, CSV, TXT"
        )
 
    # Empty file check
    size = path.stat().st_size
    if size == 0:
        logger.warning(f"[VALIDATOR] Empty file: {file_path}")
        return False, "The uploaded file is empty. Please upload a valid document."
 
    size_mb = size / (1024 * 1024)
    logger.info(f"[VALIDATOR] File OK: {path.name} ({size_mb:.2f} MB)")
    return True, "Valid"
 
 
def validate_question(question: str) -> tuple[bool, str]:
    """
    Validate and sanitize a user question.
 
    Returns:
        (True, "Valid") on success
        (False, "reason") on failure
    """
    if not question or not question.strip():
        return False, "Please enter a question."
 
    q = question.strip()
 
    if len(q) < 2:
        return False, "Question is too short. Please be more specific."
 
    if len(q) > 3000:
        return False, (
            "Question is too long (max 3000 characters). "
            "Please shorten your question."
        )
 
    # Prompt injection check
    q_lower = q.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, q_lower):
            logger.warning(f"[VALIDATOR] Possible prompt injection blocked: {q[:80]}")
            return False, (
                "Your question contains patterns that cannot be processed. "
                "Please rephrase and try again."
            )
 
    return True, "Valid"
 
 
def sanitize_text(text: str, max_length: int = 50000) -> str:
    """
    Sanitize extracted document text before processing.
    Removes null bytes, excessive whitespace, and truncates if needed.
    """
    if not text:
        return ""
 
    # Remove null bytes
    text = text.replace("\x00", "")
 
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
 
    # Collapse excessive blank lines (more than 2 consecutive)
    text = re.sub(r"\n{3,}", "\n\n", text)
 
    # Truncate if too long
    if len(text) > max_length:
        logger.warning(
            f"[VALIDATOR] Document text truncated from "
            f"{len(text)} to {max_length} chars"
        )
        text = text[:max_length]
 
    return text.strip()