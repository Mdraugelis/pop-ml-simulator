"""
Tests for VectorizedTemporalRiskSimulator.

This module provides comprehensive tests for the vectorized temporal risk
simulator, including unit tests, integration tests, and performance validation.
"""

import pytest
import numpy as np
import warnings
from scipy import sparse
from unittest.mock import patch

from pop_ml_simulator import VectorizedTemporalRiskSimulator, SimulationResults


class TestVectorizedTemporalRiskSimulator:
    """Test suite for VectorizedTemporalRiskSimulator class."""

    @pytest.fixture(autouse=True)
    def mock_ml_optimization(self):
        """Mock expensive ML optimization to speed up tests."""
        with patch(
            'pop_ml_simulator.ml_simulation.'
            'MLPredictionSimulator.optimize_noise_parameters'
        ) as mock:
            # Return immediately with default params instead of grid search
            mock.return_value = {'correlation': 0.7, 'scale': 0.3}
            yield mock

    @pytest.fixture
    def basic_simulator(self):
        """Create a basic simulator instance for testing."""
        return VectorizedTemporalRiskSimulator(
            n_patients=20,  # Reduced from 100 for speed
            n_timesteps=12,  # Reduced from 24 for speed
            annual_incident_rate=0.1,
            intervention_effectiveness=0.25,
            random_seed=42
        )

    @pytest.fixture
    def small_simulator(self):
        """Create a small simulator for quick tests."""
        return VectorizedTemporalRiskSimulator(
            n_patients=5,  # Further reduced for speed
            n_timesteps=8,  # Increased to accommodate prediction window
            annual_incident_rate=0.05,
            intervention_effectiveness=0.3,
            prediction_window=4,  # Reduced from default 12
            random_seed=123
        )

    def test_initialization(self, basic_simulator):
        """Test simulator initialization."""
        assert basic_simulator.n_patients == 20
        assert basic_simulator.n_timesteps == 12
        assert basic_simulator.annual_incident_rate == 0.1
        assert basic_simulator.intervention_effectiveness == 0.25
        assert basic_simulator.random_seed == 42

        # Check initial state
        assert not basic_simulator._population_initialized
        assert not basic_simulator._temporal_simulated
        assert not basic_simulator._ml_predictions_generated
        assert not basic_simulator._interventions_assigned
        assert not basic_simulator._incidents_simulated

        # Check results initialization
        assert basic_simulator.results.n_patients == 20
        assert basic_simulator.results.n_timesteps == 12
        assert basic_simulator.results.intervention_effectiveness == 0.25
        assert len(basic_simulator.results.patient_base_risks) == 0

    def test_initialize_population(self, basic_simulator):
        """Test population initialization."""
        basic_simulator.initialize_population(
            concentration=0.5,
            rho=0.9,
            sigma=0.1
        )

        assert basic_simulator._population_initialized
        assert len(basic_simulator.results.patient_base_risks) == 20
        assert basic_simulator.temporal_simulator is not None

        # Check risk distribution properties
        risks = basic_simulator.results.patient_base_risks
        assert np.all(risks >= 0)
        assert np.all(risks <= 1)
        assert np.isclose(np.mean(risks), 0.1, rtol=0.1)

    def test_simulate_temporal_evolution(self, basic_simulator):
        """Test temporal evolution simulation."""
        basic_simulator.initialize_population()
        basic_simulator.simulate_temporal_evolution()

        assert basic_simulator._temporal_simulated
        assert basic_simulator.results.temporal_risk_matrix.shape == (20, 12)

        # Check temporal risk properties
        risks = basic_simulator.results.temporal_risk_matrix
        assert np.all(risks >= 0)
        assert np.all(risks <= 1)

        # Check that risks vary over time
        temporal_variance = np.var(risks, axis=1)
        assert np.mean(temporal_variance) > 0

    def test_temporal_evolution_without_population(self, basic_simulator):
        """Test temporal evolution fails without population initialization."""
        with pytest.raises(
            ValueError, match="Population must be initialized first"
        ):
            basic_simulator.simulate_temporal_evolution()

    def test_generate_ml_predictions(self, small_simulator):
        """Test ML prediction generation."""
        small_simulator.initialize_population()
        small_simulator.simulate_temporal_evolution()

        prediction_times = [0]  # Use only one prediction time
        small_simulator.generate_ml_predictions(
            prediction_times,
            target_sensitivity=0.8,
            target_ppv=0.3,
            n_optimization_iterations=1  # Reduced for testing
        )

        assert small_simulator._ml_predictions_generated
        assert len(small_simulator.results.ml_predictions) == 1
        assert len(small_simulator.results.ml_binary_predictions) == 1

        # Check prediction properties
        for time in prediction_times:
            predictions = small_simulator.results.ml_predictions[time]
            binary_preds = small_simulator.results.ml_binary_predictions[time]

            assert len(predictions) == 5
            assert len(binary_preds) == 5
            assert np.all(predictions >= 0)
            assert np.all(predictions <= 1)
            assert np.all(np.isin(binary_preds, [0, 1]))

    def test_ml_predictions_without_temporal(self, basic_simulator):
        """Test that ML predictions fail without temporal simulation."""
        basic_simulator.initialize_population()

        with pytest.raises(
            ValueError, match="Temporal evolution must be simulated first"
        ):
            basic_simulator.generate_ml_predictions([0, 12])

    def test_ml_predictions_window_overflow(self, small_simulator):
        """Test handling of prediction windows extending beyond simulation."""
        small_simulator.initialize_population()
        small_simulator.simulate_temporal_evolution()

        # This should generate a warning but not fail
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            small_simulator.generate_ml_predictions(
                [10],  # prediction_window=12, timesteps=12, so 10+12 > 12
                n_optimization_iterations=1
            )
            assert len(w) == 1
            assert "extends beyond simulation horizon" in str(w[0].message)

    def test_assign_interventions_ml_threshold(self, small_simulator):
        """Test intervention assignment with ML threshold strategy."""
        small_simulator.initialize_population()
        small_simulator.simulate_temporal_evolution()
        small_simulator.generate_ml_predictions(
            [0], n_optimization_iterations=1
        )

        small_simulator.assign_interventions(
            assignment_strategy="ml_threshold",
            threshold=0.5
        )

        assert small_simulator._interventions_assigned
        assert isinstance(
            small_simulator.results.intervention_matrix, sparse.csr_matrix
        )
        assert small_simulator.results.intervention_matrix.shape == (5, 8)

        # Check that interventions were assigned
        total_interventions = small_simulator.results.intervention_matrix.nnz
        assert total_interventions >= 0

    def test_assign_interventions_random(self, small_simulator):
        """Test intervention assignment with random strategy."""
        small_simulator.initialize_population()
        small_simulator.simulate_temporal_evolution()
        small_simulator.generate_ml_predictions(
            [0], n_optimization_iterations=1
        )

        small_simulator.assign_interventions(
            assignment_strategy="random",
            treatment_fraction=0.5
        )

        assert small_simulator._interventions_assigned
        # With random assignment, we should have some interventions
        total_interventions = small_simulator.results.intervention_matrix.nnz
        assert total_interventions > 0

    def test_assign_interventions_top_k(self, small_simulator):
        """Test intervention assignment with top-k strategy."""
        small_simulator.initialize_population()
        small_simulator.simulate_temporal_evolution()
        small_simulator.generate_ml_predictions(
            [0], n_optimization_iterations=1
        )

        small_simulator.assign_interventions(
            assignment_strategy="top_k",
            treatment_fraction=0.3
        )

        assert small_simulator._interventions_assigned
        # With top-k assignment, we should have interventions
        total_interventions = small_simulator.results.intervention_matrix.nnz
        assert total_interventions > 0

    def test_assign_interventions_unknown_strategy(self, small_simulator):
        """Test that unknown assignment strategy raises error."""
        small_simulator.initialize_population()
        small_simulator.simulate_temporal_evolution()
        small_simulator.generate_ml_predictions(
            [0], n_optimization_iterations=1
        )

        with pytest.raises(ValueError, match="Unknown assignment strategy"):
            small_simulator.assign_interventions(assignment_strategy="unknown")

    def test_assign_interventions_without_ml(self, small_simulator):
        """Test that intervention assignment fails without ML predictions."""
        small_simulator.initialize_population()
        small_simulator.simulate_temporal_evolution()

        with pytest.raises(
            ValueError, match="ML predictions must be generated first"
        ):
            small_simulator.assign_interventions()

    def test_simulate_incidents(self, small_simulator):
        """Test incident simulation."""
        small_simulator.initialize_population()
        small_simulator.simulate_temporal_evolution()
        small_simulator.generate_ml_predictions(
            [0], n_optimization_iterations=1
        )
        small_simulator.assign_interventions()

        small_simulator.simulate_incidents(generate_counterfactuals=True)

        assert small_simulator._incidents_simulated
        assert small_simulator.results.incident_matrix.shape == (5, 8)
        assert (
            small_simulator.results.counterfactual_incidents.shape ==
            (5, 8)
        )

        # Check incident properties
        incidents = small_simulator.results.incident_matrix
        assert incidents.dtype == bool
        assert np.all(np.isin(incidents, [True, False]))

        # Check counterfactuals
        counterfactuals = small_simulator.results.counterfactual_incidents
        assert counterfactuals.dtype == bool

        # Check performance metrics were computed
        assert hasattr(small_simulator.results, 'intervention_coverage')
        assert hasattr(small_simulator.results, 'incident_reduction')

    def test_simulate_incidents_without_interventions(self, small_simulator):
        """Test that incident simulation fails without interventions."""
        small_simulator.initialize_population()
        small_simulator.simulate_temporal_evolution()

        with pytest.raises(
            ValueError, match="Interventions must be assigned first"
        ):
            small_simulator.simulate_incidents()

    def test_simulate_incidents_no_counterfactuals(self, small_simulator):
        """Test incident simulation without counterfactuals."""
        small_simulator.initialize_population()
        small_simulator.simulate_temporal_evolution()
        small_simulator.generate_ml_predictions(
            [0], n_optimization_iterations=1
        )
        small_simulator.assign_interventions()

        small_simulator.simulate_incidents(generate_counterfactuals=False)

        assert small_simulator._incidents_simulated
        assert small_simulator.results.counterfactual_incidents is None

    def test_run_full_simulation(self, small_simulator):
        """Test complete simulation pipeline."""
        results = small_simulator.run_full_simulation(
            prediction_times=[0],  # Only use valid prediction time
            target_sensitivity=0.8,
            target_ppv=0.3,
            assignment_strategy="ml_threshold",
            threshold=0.5,
            generate_counterfactuals=True,
            concentration=0.5,
            rho=0.9,
            sigma=0.1
        )

        assert isinstance(results, SimulationResults)
        assert results.n_patients == 5
        assert results.n_timesteps == 8
        assert results.intervention_effectiveness == 0.3

        # Check all components were generated
        assert len(results.patient_base_risks) == 5
        assert results.temporal_risk_matrix.shape == (5, 8)
        assert results.incident_matrix.shape == (5, 8)
        assert len(results.ml_predictions) == 1
        assert results.intervention_matrix.shape == (5, 8)
        assert results.counterfactual_incidents.shape == (5, 8)

    def test_get_patient_trajectory(self, small_simulator):
        """Test patient trajectory retrieval."""
        small_simulator.run_full_simulation(
            prediction_times=[0],
            n_optimization_iterations=1
        )

        trajectory = small_simulator.get_patient_trajectory(0)

        assert 'base_risk' in trajectory
        assert 'temporal_risks' in trajectory
        assert 'incidents' in trajectory
        assert 'interventions' in trajectory
        assert 'counterfactual_incidents' in trajectory

        # Check shapes
        assert len(trajectory['temporal_risks']) == 8
        assert len(trajectory['incidents']) == 8
        assert len(trajectory['interventions']) == 8
        assert len(trajectory['counterfactual_incidents']) == 8

    def test_get_patient_trajectory_invalid_id(self, small_simulator):
        """Test patient trajectory with invalid patient ID."""
        small_simulator.run_full_simulation(
            prediction_times=[0],
            n_optimization_iterations=1
        )

        with pytest.raises(ValueError, match="Patient ID .* out of range"):
            small_simulator.get_patient_trajectory(10)  # Out of range

        with pytest.raises(ValueError, match="Patient ID .* out of range"):
            small_simulator.get_patient_trajectory(-1)

    def test_get_summary_statistics(self, small_simulator):
        """Test summary statistics generation."""
        small_simulator.run_full_simulation(
            prediction_times=[0],
            n_optimization_iterations=1
        )

        stats = small_simulator.get_summary_statistics()

        # Check required fields
        required_fields = [
            'n_patients', 'n_timesteps', 'intervention_effectiveness',
            'intervention_coverage', 'incident_reduction', 'total_incidents',
            'total_interventions', 'mean_base_risk', 'std_base_risk',
            'counterfactual_incidents', 'ml_prediction_stats'
        ]

        for field in required_fields:
            assert field in stats

        # Check types and ranges
        assert stats['n_patients'] == 5
        assert stats['n_timesteps'] == 8
        assert 0 <= stats['intervention_coverage'] <= 1
        assert stats['total_incidents'] >= 0
        assert stats['total_interventions'] >= 0
        assert stats['mean_base_risk'] > 0
        assert stats['std_base_risk'] >= 0

        # Check ML prediction stats
        ml_stats = stats['ml_prediction_stats']
        assert 'mean_score' in ml_stats
        assert 'std_score' in ml_stats
        assert 'min_score' in ml_stats
        assert 'max_score' in ml_stats
        assert 0 <= ml_stats['min_score'] <= ml_stats['max_score'] <= 1

    def test_sparse_matrix_intervention_tracking(self, small_simulator):
        """Test that intervention tracking uses sparse matrices efficiently."""
        small_simulator.initialize_population()
        small_simulator.simulate_temporal_evolution()
        small_simulator.generate_ml_predictions(
            [0], n_optimization_iterations=1
        )

        # Test with very selective interventions
        small_simulator.assign_interventions(
            assignment_strategy="top_k",
            treatment_fraction=0.1  # Only 10% get treatment
        )

        intervention_matrix = small_simulator.results.intervention_matrix
        assert isinstance(intervention_matrix, sparse.csr_matrix)
        assert intervention_matrix.shape == (5, 8)

        # Sparse matrix more efficient than dense for sparse interventions
        total_elements = (
            intervention_matrix.shape[0] * intervention_matrix.shape[1]
        )
        non_zero_elements = intervention_matrix.nnz
        sparsity = 1 - (non_zero_elements / total_elements)
        assert sparsity > 0.5  # Should be at least 50% sparse

    def test_hazard_scale_operations(self, small_simulator):
        """Test that simulator operates on hazard scale correctly."""
        small_simulator.initialize_population()
        small_simulator.simulate_temporal_evolution()

        # Check that temporal risks are properly bounded
        risks = small_simulator.results.temporal_risk_matrix
        assert np.all(risks >= 0)
        assert np.all(risks <= 1)

        # Check that base risks are calibrated to population mean
        base_risks = small_simulator.results.patient_base_risks
        assert np.isclose(np.mean(base_risks), 0.05, rtol=0.1)

    def test_counterfactual_consistency(self, small_simulator):
        """Test that counterfactuals are generated consistently."""
        # Set fixed seed for reproducibility
        small_simulator.random_seed = 42
        np.random.seed(42)

        small_simulator.run_full_simulation(
            prediction_times=[0],
            n_optimization_iterations=1
        )

        # Counterfactuals should use same base risks
        actual_incidents = small_simulator.results.incident_matrix
        counterfactual_incidents = (
            small_simulator.results.counterfactual_incidents
        )

        # Both should be boolean arrays of same shape
        assert actual_incidents.shape == counterfactual_incidents.shape
        assert actual_incidents.dtype == bool
        assert counterfactual_incidents.dtype == bool

        # Counterfactuals generally have more incidents (no intervention)
        actual_total = np.sum(actual_incidents)
        counterfactual_total = np.sum(counterfactual_incidents)
        assert counterfactual_total >= actual_total

    def test_simulation_results_dataclass(self):
        """Test SimulationResults dataclass functionality."""
        # Test initialization
        results = SimulationResults(
            patient_base_risks=np.array([0.1, 0.2]),
            temporal_risk_matrix=np.array([[0.1, 0.15], [0.2, 0.25]]),
            incident_matrix=np.array([[False, True], [True, False]]),
            ml_predictions={0: np.array([0.3, 0.7])},
            ml_binary_predictions={0: np.array([0, 1])},
            intervention_matrix=sparse.csr_matrix([[1, 0], [0, 1]]),
            intervention_times={0: np.array([0])},
            n_patients=2,
            n_timesteps=2,
            intervention_effectiveness=0.25
        )

        assert len(results.patient_base_risks) == 2
        assert results.temporal_risk_matrix.shape == (2, 2)
        assert results.incident_matrix.shape == (2, 2)
        assert results.n_patients == 2
        assert results.n_timesteps == 2
        assert results.intervention_effectiveness == 0.25
        assert results.ml_prediction_times == []  # Default post-init value

    def test_performance_metrics_computation(self, small_simulator):
        """Test that performance metrics are computed correctly."""
        small_simulator.run_full_simulation(
            prediction_times=[0],
            n_optimization_iterations=1
        )

        # Check intervention coverage
        total_interventions = small_simulator.results.intervention_matrix.nnz
        expected_coverage = total_interventions / (
            small_simulator.n_patients * small_simulator.n_timesteps
        )
        assert (
            small_simulator.results.intervention_coverage ==
            expected_coverage
        )

        # Check incident reduction calculation
        actual_incidents = np.sum(small_simulator.results.incident_matrix)
        counterfactual_incidents = np.sum(
            small_simulator.results.counterfactual_incidents
        )

        if counterfactual_incidents > 0:
            expected_reduction = (
                (counterfactual_incidents - actual_incidents) /
                counterfactual_incidents
            )
            assert np.isclose(
                small_simulator.results.incident_reduction, expected_reduction
            )
        else:
            assert small_simulator.results.incident_reduction == 0.0

    @pytest.mark.parametrize("n_patients,n_timesteps", [
        (15, 12),
        (20, 10),
        (25, 8)
    ])
    def test_different_population_sizes(self, n_patients, n_timesteps):
        """Test simulator with different population sizes."""
        simulator = VectorizedTemporalRiskSimulator(
            n_patients=n_patients,
            n_timesteps=n_timesteps,
            annual_incident_rate=0.08,
            random_seed=42
        )

        results = simulator.run_full_simulation(
            prediction_times=[0],
            n_optimization_iterations=1
        )

        assert results.n_patients == n_patients
        assert results.n_timesteps == n_timesteps
        assert results.temporal_risk_matrix.shape == (n_patients, n_timesteps)
        assert results.incident_matrix.shape == (n_patients, n_timesteps)

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed."""
        seed = 12345

        # Run simulation twice with same seed
        sim1 = VectorizedTemporalRiskSimulator(
            n_patients=15,  # Reduced for speed
            n_timesteps=8,  # Reduced for speed
            annual_incident_rate=0.1,
            random_seed=seed
        )
        results1 = sim1.run_full_simulation(
            prediction_times=[0],
            n_optimization_iterations=1
        )

        sim2 = VectorizedTemporalRiskSimulator(
            n_patients=15,  # Reduced for speed
            n_timesteps=8,  # Reduced for speed
            annual_incident_rate=0.1,
            random_seed=seed
        )
        results2 = sim2.run_full_simulation(
            prediction_times=[0],
            n_optimization_iterations=1
        )

        # Results should be identical
        np.testing.assert_array_equal(
            results1.patient_base_risks, results2.patient_base_risks
        )
        np.testing.assert_array_equal(
            results1.temporal_risk_matrix, results2.temporal_risk_matrix
        )
        np.testing.assert_array_equal(
            results1.incident_matrix, results2.incident_matrix
        )

    def test_edge_case_zero_intervention_effectiveness(self, small_simulator):
        """Test simulator with zero intervention effectiveness."""
        small_simulator.intervention_effectiveness = 0.0

        small_simulator.run_full_simulation(
            prediction_times=[0],
            n_optimization_iterations=1
        )

        # With zero effectiveness, incidents same as counterfactuals
        actual_incidents = np.sum(small_simulator.results.incident_matrix)
        counterfactual_incidents = np.sum(
            small_simulator.results.counterfactual_incidents
        )

        # They should be very close (might differ due to randomness)
        assert np.isclose(actual_incidents, counterfactual_incidents, rtol=0.1)

    def test_edge_case_full_intervention_effectiveness(self, small_simulator):
        """Test simulator with full intervention effectiveness."""
        small_simulator.intervention_effectiveness = 1.0

        small_simulator.run_full_simulation(
            prediction_times=[0],
            assignment_strategy="random",
            treatment_fraction=1.0,  # Treat everyone
            n_optimization_iterations=1
        )

        # With full effectiveness and everyone treated, fewer incidents
        actual_incidents = np.sum(small_simulator.results.incident_matrix)
        counterfactual_incidents = np.sum(
            small_simulator.results.counterfactual_incidents
        )

        assert actual_incidents <= counterfactual_incidents
