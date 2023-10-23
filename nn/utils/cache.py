from functools import lru_cache, wraps
from datetime import datetime, timedelta


def timed_lru_cache(seconds: int, maxsize: int = 128):
    """Support Lifetime cache (in seconds) and maximum size of cache for LRU cache.
    Check the cachetools module (github.com/cachetools/) for more various memorizing collections and decorators.

    Args:
        seconds (int): define period for function to be cached in seconds.
        maxsize (int, optional): define maximum number of entries before cache evicts old items. Defaults to 128.
    """

    def wrapper_cache(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = timedelta(seconds=seconds)
        func.expiration = datetime.now() + func.lifetime

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            # before accessing an entry in the cache, check whether the current date is past expiration date.
            # if so, clear the cache and recompute the lifetime and expiration date.
            if datetime.now() >= func.expiration:
                func.cache_clear()
                func.expiration = datetime.now() + func.lifetime
            return func(*args, **kwargs)

        return wrapped_func

    return wrapper_cache
