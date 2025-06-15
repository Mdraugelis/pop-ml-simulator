from src.utils.logging import log_call


@log_call
def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b
