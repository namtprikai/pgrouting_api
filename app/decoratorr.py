import time
from functools import wraps

def timeit(name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            print(f"[Timing] {name}: {time.time() - start:.2f}s")
            return result
        return wrapper
    return decorator