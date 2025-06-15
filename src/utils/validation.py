from omegaconf import DictConfig

from src.utils.logging import log_call


@log_call
def validate_config(cfg: DictConfig) -> None:
    """Simple validation for simulation configs."""

    if cfg.population.n_persons <= 0:
        raise ValueError("n_persons must be positive")
    if cfg.population.age_min >= cfg.population.age_max:
        raise ValueError("age_min must be less than age_max")
    if cfg.hospitals.n_hospitals <= 0:
        raise ValueError("n_hospitals must be positive")
    if not 0 <= cfg.hazards.annual_incident_rate <= 1:
        raise ValueError("annual_incident_rate must be between 0 and 1")
