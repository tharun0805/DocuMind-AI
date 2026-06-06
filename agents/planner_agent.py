import os
from loguru import logger

DATAFRAME_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def plan_route(intent: str, file_path: str = "") -> str:
    """
    CSV/Excel → always dataframe (pandas queries all intents).
    All other files → retrieval + QA.
    """
    logger.debug(f"Planning route — intent={intent} file={os.path.basename(file_path)}")
    if file_path:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in DATAFRAME_EXTENSIONS:
            logger.debug("Route: dataframe")
            return "dataframe"
    logger.debug("Route: retrieval")
    return "retrieval"