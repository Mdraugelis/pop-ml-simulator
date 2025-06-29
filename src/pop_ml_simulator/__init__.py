"""Population-based Machine Learning Simulator package."""

from typing import List

from .sample import add
from .risk_distribution import assign_patient_risks, simulate_annual_incidents
from .temporal_dynamics import (
    TemporalRiskSimulator,
    EnhancedTemporalRiskSimulator,
    simulate_ar1_process
)
from .hazard_modeling import (
    annual_risk_to_hazard,
    hazard_to_timestep_probability,
    IncidentGenerator,
    CompetingRiskIncidentGenerator
)

__all__: List[str] = [
    "add",
    # Risk distribution functions
    "assign_patient_risks",
    "simulate_annual_incidents",
    # Temporal dynamics classes and functions
    "TemporalRiskSimulator",
    "EnhancedTemporalRiskSimulator",
    "simulate_ar1_process",
    # Hazard modeling functions and classes
    "annual_risk_to_hazard",
    "hazard_to_timestep_probability",
    "IncidentGenerator",
    "CompetingRiskIncidentGenerator",
]
__version__ = "0.1.0"
