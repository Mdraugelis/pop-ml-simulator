import pytest
import sys


def main() -> None:
    args = ["-vv", "--cov=src", "--cov-report=term-missing"] + sys.argv[1:]
    raise SystemExit(pytest.main(args))


if __name__ == "__main__":
    main()
