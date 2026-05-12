from loguru import logger
from pathlib import Path


def get_logger(name: str):
    Path("logs").mkdir(exist_ok=True)

    logger.add(
        "logs/documind.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
    )

    return logger.bind(name=name)