"""
Shared test fixtures for performance optimization.

This module provides shared fixtures to reduce test setup time and
avoid redundant computations across test modules.
"""

import os
import sys

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pop_ml_simulator.risk_distribution import (  # noqa: E402
    assign_patient_risks
)
from pop_ml_simulator.temporal_dynamics import (  # noqa: E402
    build_temporal_risk_matrix
)
from pop_ml_simulator.hazard_modeling import IncidentGenerator  # noqa: E402


@pytest.fixture(scope="session")
def small_patient_risks():
    """Small dataset of patient risks (100 patients) for fast tests."""
    np.random.seed(42)
    return assign_patient_risks(
        n_patients=100,
        annual_incident_rate=0.1,
        concentration=0.5,
        random_seed=42
    )


@pytest.fixture(scope="session")
def medium_patient_risks():
    """Medium dataset of patient risks (500 patients) for most tests."""
    np.random.seed(42)
    return assign_patient_risks(
        n_patients=500,
        annual_incident_rate=0.1,
        concentration=0.5,
        random_seed=42
    )


@pytest.fixture(scope="session")
def large_patient_risks():
    """Large dataset of patient risks (1000 patients) for integration tests."""
    np.random.seed(42)
    return assign_patient_risks(
        n_patients=1000,
        annual_incident_rate=0.1,
        concentration=0.5,
        random_seed=42
    )


@pytest.fixture(scope="session")
def small_temporal_matrix(small_patient_risks):
    """Small temporal risk matrix (100 patients × 12 timesteps)."""
    np.random.seed(42)
    return build_temporal_risk_matrix(
        small_patient_risks,
        n_timesteps=12,
        random_seed=42
    )


@pytest.fixture(scope="session")
def medium_temporal_matrix(medium_patient_risks):
    """Medium temporal risk matrix (500 patients × 26 timesteps)."""
    np.random.seed(42)
    return build_temporal_risk_matrix(
        medium_patient_risks,
        n_timesteps=26,
        random_seed=42
    )


@pytest.fixture(scope="session")
def large_temporal_matrix(large_patient_risks):
    """Large temporal risk matrix (1000 patients × 52 timesteps)."""
    np.random.seed(42)
    return build_temporal_risk_matrix(
        large_patient_risks,
        n_timesteps=52,
        random_seed=42
    )


@pytest.fixture(scope="session")
def incident_generator():
    """Shared incident generator instance."""
    return IncidentGenerator(timestep_duration=1/52)


@pytest.fixture(scope="session")
def small_incident_labels(small_patient_risks, incident_generator):
    """Pre-computed incident labels for small dataset."""
    np.random.seed(42)
    true_labels = np.zeros(len(small_patient_risks), dtype=int)

    for _ in range(12):  # 12 weeks
        incidents = incident_generator.generate_incidents(small_patient_risks)
        true_labels |= incidents

    return true_labels


@pytest.fixture(scope="session")
def medium_incident_labels(medium_patient_risks, incident_generator):
    """Pre-computed incident labels for medium dataset."""
    np.random.seed(42)
    true_labels = np.zeros(len(medium_patient_risks), dtype=int)

    for _ in range(26):  # 26 weeks
        incidents = incident_generator.generate_incidents(medium_patient_risks)
        true_labels |= incidents

    return true_labels


@pytest.fixture(scope="session")
def large_incident_labels(large_patient_risks, incident_generator):
    """Pre-computed incident labels for large dataset."""
    np.random.seed(42)
    true_labels = np.zeros(len(large_patient_risks), dtype=int)

    for _ in range(26):  # 26 weeks
        incidents = incident_generator.generate_incidents(large_patient_risks)
        true_labels |= incidents

    return true_labels


@pytest.fixture(scope="session")
def cached_ml_params():
    """Pre-optimized ML parameters to avoid repeated optimization."""
    return {
        'correlation': 0.75,
        'scale': 0.25,
        'target_sensitivity': 0.8,
        'target_ppv': 0.3
    }
