import sys
import os
from pathlib import Path
from loguru import logger as _logger
 
# ── Ensure logs directory exists ──────────────────────────────────
Path("logs").mkdir(exist_ok=True)
 
# ── Remove default Loguru handler ─────────────────────────────────
_logger.remove()
 
# ── Console handler (WARNING and above only) ──────────────────────
_logger.add(
    sys.stderr,
    level="WARNING",
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[module]}</cyan> | "
        "<level>{message}</level>"
    ),
    colorize=True,
)
 
# ── Main log file (DEBUG and above, rotated) ──────────────────────
_logger.add(
    "logs/documind.log",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    format=(
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{extra[module]: <30} | "
        "{message}"
    ),
    encoding="utf-8",
    enqueue=True,       # thread-safe
    backtrace=True,     # full traceback on exceptions
    diagnose=False,     # don't expose local variables in prod
)
 
# ── Error-only log file ────────────────────────────────────────────
_logger.add(
    "logs/errors.log",
    level="ERROR",
    rotation="5 MB",
    retention="14 days",
    compression="zip",
    format=(
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{extra[module]: <30} | "
        "{message}\n{exception}"
    ),
    encoding="utf-8",
    enqueue=True,
    backtrace=True,
    diagnose=False,
)
 
 
def get_logger(module_name: str):
    """
    Return a logger bound to a specific module name.
 
    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Message here")
    """
    return _logger.bind(module=module_name)
 
 
# Default logger for backward compatibility
logger = _logger.bind(module="app")