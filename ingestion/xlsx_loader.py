import openpyxl
from loguru import logger


def load_xlsx(file_path: str) -> str:
    workbook = openpyxl.load_workbook(file_path)
    text = ""

    for sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
        text += f"Sheet: {sheet_name}\n"

        for row in sheet.iter_rows(values_only=True):
            row_text = " | ".join(
                str(cell) for cell in row if cell is not None
            )
            if row_text.strip():
                text += row_text + "\n"

    logger.info(f"XLSX loaded: {file_path}")
    return text