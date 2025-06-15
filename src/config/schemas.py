from dataclasses import dataclass
from typing import Dict


@dataclass
class PopulationConfig:
    n_persons: int
    age_min: int
    age_max: int
    random_seed: int


@dataclass
class HospitalConfig:
    n_hospitals: int
    size_distribution: Dict[str, float]
    quality_range: Dict[str, float]


@dataclass
class SimulationConfig:
    total_months: int
    delta_t: float = 1.0
    checkpoint_frequency: int = 1


@dataclass
class HazardsConfig:
    annual_incident_rate: float
    base_mortality_hazard: float
    incident_mortality_multiplier: float


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    output_dir: str
