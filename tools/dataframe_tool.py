import pandas as pd
from loguru import logger


def load_dataframe(file_path: str) -> pd.DataFrame:
    extension = file_path.split(".")[-1].lower()

    if extension == "csv":
        df = pd.read_csv(file_path)
        logger.info(f"CSV loaded into DataFrame: {file_path}")

    elif extension in ["xlsx", "xls"]:
        df = pd.read_excel(file_path)
        logger.info(f"Excel loaded into DataFrame: {file_path}")

    else:
        logger.error(f"Unsupported file type for DataFrame: {extension}")
        raise ValueError(f"Unsupported file type: {extension}")

    logger.info(f"DataFrame shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def get_dataframe_info(df: pd.DataFrame) -> str:
    info = f"Columns: {list(df.columns)}\n"
    info += f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
    info += f"Data Types:\n{df.dtypes.to_string()}\n"
    info += f"\nFirst 5 rows:\n{df.head().to_string()}"
    return info