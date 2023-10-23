import functools
import time
import inspect
import logging

logger = logging.getLogger(__name__)


def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Do something before func
        value = func(*args, **kwargs)
        # Do something after func
        return value
    return wrapper


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        run_time = time.perf_counter() - start_time
        logger.info(f"Finish running {func.__name__!r} in {run_time:.4f} sec")
        return value
    return wrapper_timer


def debug(func):
    """Log function signature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        signature = f"{func.__name__}{inspect.signature(func)}"
        logger.debug(f"Calling {signature}")
        value = func(*args, **kwargs)
        logger.debug(f"{func.__name__!r} returns {value!r}")
        return value
    return wrapper_debug