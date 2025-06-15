from itertools import product

import pytest

from src.config import load_config
from src.utils.validation import validate_config
from tests.fixtures.sample_configs import make_invalid_config


@pytest.mark.parametrize(
    "overrides",
    [
        [
            f"population={p}",
            f"hospitals={h}",
            f"simulation={s}",
            f"hazards={z}",
        ]
        for p, h, s, z in product(
            ["small", "medium", "large"],
            ["basic", "extended"],
            ["short", "long"],
            ["low_risk", "standard"],
        )
    ],
)
def test_all_configs_load(overrides):
    cfg = load_config(overrides)
    assert cfg is not None


def test_validation_fails_on_invalid():
    cfg = make_invalid_config()
    with pytest.raises(ValueError):
        validate_config(cfg)
