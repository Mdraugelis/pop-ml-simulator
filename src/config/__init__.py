from pathlib import Path
from typing import List, Optional

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src.utils.logging import log_call
from src.utils.validation import validate_config

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@log_call
def load_config(overrides: Optional[List[str]] = None) -> DictConfig:
    """Load and validate a configuration using Hydra."""

    overrides = overrides or []
    with initialize_config_dir(
        CONFIG_DIR.resolve().as_posix(), version_base=None
    ):
        cfg = compose(config_name="config", overrides=overrides)
    validate_config(cfg)
    return cfg
