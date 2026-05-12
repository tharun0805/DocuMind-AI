from docx import Document
from loguru import logger


def load_docx(file_path: str) -> str:
    doc = Document(file_path)
    text = ""

    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text += paragraph.text + "\n"

    logger.info(f"DOCX loaded: {file_path}")
    return text