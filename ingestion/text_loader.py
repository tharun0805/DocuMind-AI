from loguru import logger


def load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    logger.info(f"TXT loaded: {file_path}")
    return text