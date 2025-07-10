import functools
import logging
import os
import time
from typing import Any, Callable, TypeVar

# Default to CRITICAL (effectively off) unless explicitly set for debug
LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "CRITICAL").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.CRITICAL))

F = TypeVar("F", bound=Callable[..., Any])


def _do_redact(value: Any) -> str:
    """Internal implementation used to sanitize values."""
    return "<redacted>"


def log_call(func: F) -> F:
    """Decorator that logs function entry, exit and runtime at DEBUG level."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = logging.getLogger(func.__module__)
        log_debug = logger.isEnabledFor(logging.DEBUG)
        if log_debug:
            logger.debug("Entering %s", func.__qualname__)
            sanitized_args = [_do_redact(a) for a in args]
            sanitized_kwargs = {k: _do_redact(v) for k, v in kwargs.items()}
            logger.debug("args=%s kwargs=%s", sanitized_args, sanitized_kwargs)
        start = time.time()
        result = func(*args, **kwargs)
        runtime_ms = (time.time() - start) * 1000.0
        if log_debug:
            logger.debug("return=%s", _do_redact(result))
            logger.debug("Exiting %s (%.2fms)", func.__qualname__, runtime_ms)
        return result

    return wrapper  # type: ignore[return-value]


@log_call
def redact(value: Any) -> str:
    """Redact sensitive information from logs."""
    return _do_redact(value)
