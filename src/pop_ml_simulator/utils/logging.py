import time
import logging
from functools import wraps

logging.basicConfig(level=logging.INFO)

def log_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.info(f"Calling: {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Finished: {func.__name__} in {end_time - start_time:.2f}s")
        return result
    return wrapper
