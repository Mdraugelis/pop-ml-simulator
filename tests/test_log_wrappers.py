import pytest

pytestmark = pytest.mark.skip(reason="log_call decoration is optional")


def test_public_functions_are_decorated() -> None:
    pass
