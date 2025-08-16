import argparse
import os
import sys

import pytest

# Add src directory to Python path so tests can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run test suites")
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Include integration tests (may be slow)",
    )
    parsed_args, remaining = parser.parse_known_args()

    try:
        import pytest_cov  # noqa: F401
        args = ["-vv", "--cov=src", "--cov-report=term-missing"]
    except ImportError:
        args = ["-vv"]

    if not parsed_args.integration:
        args += ["-m", "not integration"]

    args += remaining
    raise SystemExit(pytest.main(args))


if __name__ == "__main__":
    main()
