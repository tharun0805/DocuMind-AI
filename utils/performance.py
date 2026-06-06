import time
from functools import wraps
from typing import Dict, Optional
from loguru import logger
 
 
# ── Decorator ─────────────────────────────────────────────────────
 
def timer(func):
    """
    Decorator — logs execution time of any function.
 
    Usage:
        @timer
        def my_function(): ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        level = "WARNING" if duration > 10 else "INFO"
        logger.log(level, f"[TIMER] {func.__name__} completed in {duration:.3f}s")
        return result
    return wrapper
 
 
# ── Step Tracker ──────────────────────────────────────────────────
 
class PerformanceTracker:
    """
    Track timing for multi-step operations.
 
    Usage:
        tracker = PerformanceTracker()
        tracker.start("reading")
        # ... do reading ...
        tracker.end("reading")
        tracker.start("indexing")
        # ... do indexing ...
        tracker.end("indexing")
        tracker.report()
    """
 
    SLOW_THRESHOLD_SECONDS = 10.0
 
    def __init__(self, operation_name: str = "operation"):
        self._name = operation_name
        self._starts: Dict[str, float] = {}
        self._durations: Dict[str, float] = {}
        self._total_start = time.perf_counter()
 
    def start(self, step: str) -> None:
        self._starts[step] = time.perf_counter()
        logger.debug(f"[PERF] [{self._name}] Starting: {step}")
 
    def end(self, step: str) -> float:
        if step not in self._starts:
            logger.warning(f"[PERF] end() called for unstarted step: {step}")
            return 0.0
        duration = time.perf_counter() - self._starts[step]
        self._durations[step] = duration
        level = "WARNING" if duration > self.SLOW_THRESHOLD_SECONDS else "INFO"
        logger.log(level, f"[PERF] [{self._name}] {step}: {duration:.3f}s"
                   + (" [SLOW]" if duration > self.SLOW_THRESHOLD_SECONDS else ""))
        return duration
 
    def report(self) -> Dict[str, float]:
        total = time.perf_counter() - self._total_start
        self._durations["__total__"] = total
 
        slow_steps = [
            s for s, d in self._durations.items()
            if d > self.SLOW_THRESHOLD_SECONDS and s != "__total__"
        ]
 
        logger.info(
            f"[PERF] [{self._name}] Summary — "
            f"Total: {total:.3f}s | "
            f"Steps: {len(self._durations) - 1} | "
            f"Slow: {slow_steps or 'none'}"
        )
        return self._durations
 
    def get(self, step: str) -> Optional[float]:
        return self._durations.get(step)