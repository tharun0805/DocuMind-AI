import pandas as pd
from loguru import logger


def load_csv(file_path: str) -> str:
    df = pd.read_csv(file_path)
    text = df.to_string(index=False)

    logger.info(f"CSV loaded: {file_path}")
    return text