import pytest
import sys
import os

# Add src directory to Python path so tests can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def main() -> None:
    # Set log level to suppress logging output during tests
    os.environ['APP_LOG_LEVEL'] = 'ERROR'
    
    # Only add coverage args if pytest-cov is available
    try:
        import pytest_cov  # noqa: F401
        args = ["-vv", "--cov=src", "--cov-report=term-missing"] + sys.argv[1:]
    except ImportError:
        args = ["-vv"] + sys.argv[1:]
    raise SystemExit(pytest.main(args))


if __name__ == "__main__":
    main()
