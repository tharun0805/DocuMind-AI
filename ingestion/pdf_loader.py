import pdfplumber
from loguru import logger


def load_pdf(file_path: str) -> str:
    text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    logger.info(f"PDF loaded: {file_path}")
    return text