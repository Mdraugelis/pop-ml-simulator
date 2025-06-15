from omegaconf import OmegaConf


def make_invalid_config() -> OmegaConf:
    """Return a config with invalid population size."""

    cfg = OmegaConf.create(
        {
            "population": {
                "n_persons": -1,
                "age_min": 10,
                "age_max": 20,
                "random_seed": 1,
            },
            "hospitals": {
                "n_hospitals": 1,
                "size_distribution": {"small": 1.0},
                "quality_range": {"min": 0.5, "max": 0.9},
            },
            "simulation": {
                "total_months": 1,
                "delta_t": 1.0,
                "checkpoint_frequency": 1,
            },
            "hazards": {
                "annual_incident_rate": 0.01,
                "base_mortality_hazard": 0.001,
                "incident_mortality_multiplier": 1.0,
            },
        }
    )
    return cfg
