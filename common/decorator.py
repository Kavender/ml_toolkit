import functools
Import time

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
        print(f"Finish running {func.__name__!r} in {run_time:.4f} sec")
        return value
    return wrapper_timer


def debug(func):
    "Print function signature and return vaue"
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k} = {v!r}" for k, v in kwargs.items()]
        print(f"Calling {func.__name__} ({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returns {value!r}")
        return value
    return wrapper_debug
