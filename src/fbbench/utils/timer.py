import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def time_block(description: str) -> Iterator[None]:
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        duration = end - start
        print(f"[TIMER] {description}: {duration:.2f}s")

