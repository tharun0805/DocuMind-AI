import os
from loguru import logger

from ingestion.pdf_loader import load_pdf
from ingestion.docx_loader import load_docx
from ingestion.pptx_loader import load_pptx
from ingestion.xlsx_loader import load_xlsx
from ingestion.csv_loader import load_csv
from ingestion.text_loader import load_txt


def load_document(file_path: str) -> str:
    extension = os.path.splitext(file_path)[1].lower()

    loaders = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".pptx": load_pptx,
        ".xlsx": load_xlsx,
        ".csv": load_csv,
        ".txt": load_txt,
    }

    if extension not in loaders:
        logger.error(f"Unsupported file type: {extension}")
        raise ValueError(f"Unsupported file type: {extension}")

    logger.info(f"Loading document: {file_path}")
    return loaders[extension](file_path)