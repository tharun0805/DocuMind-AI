import time
from loguru import logger
from functools import wraps


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        duration = end - start
        logger.info(f"⏱️ {func.__name__} completed in {duration:.2f}s")
        return result
    return wrapper


class PerformanceTracker:
    def __init__(self):
        self.timings = {}

    def start(self, step: str):
        self.timings[step] = time.time()
        logger.info(f"⏱️ Starting: {step}")

    def end(self, step: str):
        if step in self.timings:
            duration = time.time() - self.timings[step]
            logger.info(f"✅ Completed: {step} in {duration:.2f}s")
            return duration
        return 0

    def get_summary(self) -> str:
        summary = "Performance Summary:\n"
        for step, start_time in self.timings.items():
            summary += f"  {step}: recorded\n"
        return summary