import time
from typing import Callable, Any

def retry_backoff(fn: Callable[[], Any], retries: int = 3, base_delay: float = 1.0):
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:
            attempt += 1
            if attempt >= retries:
                raise
            time.sleep(base_delay * (2 ** (attempt - 1)))
