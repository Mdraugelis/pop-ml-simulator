"""
Tests for causal inference methods.
"""

import os
import sys
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pop_ml_simulator import VectorizedTemporalRiskSimulator
from pop_ml_simulator.causal_inference import RDDAnalysis, DiDAnalysis, ITSAnalysis


def test_rdd_analysis():
    """Test RDD analysis."""
    simulator = VectorizedTemporalRiskSimulator(
        n_patients=1000,
        n_timesteps=12,
        annual_incident_rate=0.1,
        intervention_effectiveness=0.25,
        random_seed=42
    )
    results = simulator.run_full_simulation(
        prediction_times=[0],
        assignment_strategy="ml_threshold",
        threshold=0.5
    )
    rdd_analyzer = RDDAnalysis(results, risk_threshold=0.5, bandwidth=0.1)
    rdd_results = rdd_analyzer.analyze_intervention_effect()
    assert "estimated_effect" in rdd_results
    assert np.isclose(rdd_results['true_effect'], 0.25)


def test_did_analysis():
    """Test DiD analysis."""
    simulator = VectorizedTemporalRiskSimulator(
        n_patients=1000,
        n_timesteps=48,
        annual_incident_rate=0.1,
        intervention_effectiveness=0.25,
        random_seed=42
    )
    results = simulator.run_full_simulation(
        prediction_times=[0, 24],
        assignment_strategy="random",
        treatment_fraction=0.5
    )
    did_analyzer = DiDAnalysis(results, intervention_start_time=24)
    did_results = did_analyzer.analyze_intervention_effect()
    assert "estimated_effect" in did_results
    assert np.isclose(did_results['true_effect'], 0.25)


def test_its_analysis():
    """Test ITS analysis."""
    simulator = VectorizedTemporalRiskSimulator(
        n_patients=1000,
        n_timesteps=48,
        annual_incident_rate=0.1,
        intervention_effectiveness=0.25,
        random_seed=42
    )
    # In ITS, the intervention is applied to the whole population at a specific time
    # This requires a slightly different setup than the standard run_full_simulation
    simulator.initialize_population()
    simulator.simulate_temporal_evolution()
    simulator.generate_ml_predictions(prediction_times=[]) # No ML predictions needed for this ITS

    # Manually create a population-wide intervention
    intervention_matrix = np.zeros((simulator.n_patients, simulator.n_timesteps))
    intervention_matrix[:, 24:] = 1
    simulator.results.intervention_matrix = intervention_matrix
    simulator._interventions_assigned = True

    simulator.simulate_incidents()

    its_analyzer = ITSAnalysis(simulator.results, intervention_time=24)
    its_results = its_analyzer.analyze_intervention_effect()
    assert "relative_effect" in its_results
    assert np.isclose(its_results['true_effect'], 0.25)
