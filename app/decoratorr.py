import time
import asyncio
from functools import wraps

def timeit(name):
    """
    A decorator that measures the execution time of a function.
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            # Async version
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start = time.time()
                result = await func(*args, **kwargs)
                print(f"[Timing] {name}: {time.time() - start:.2f}s")
                return result
            return wrapper
        else:
            # Sync version
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                print(f"[Timing] {name}: {time.time() - start:.2f}s")
                return result
            return wrapper
    return decorator
