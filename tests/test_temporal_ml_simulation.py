"""
Tests for temporal ML simulation functionality.

This module tests the temporal-aware ML prediction generation features
that integrate risk trajectories over prediction windows.
"""

import unittest
import numpy as np
import pandas as pd
import pytest

from pop_ml_simulator.ml_simulation import (
    MLPredictionSimulator,
    generate_temporal_ml_predictions,
    validate_temporal_sensitivity,
    benchmark_temporal_ml_performance
)
from pop_ml_simulator.temporal_dynamics import build_temporal_risk_matrix
from pop_ml_simulator.risk_distribution import assign_patient_risks


class TestTemporalMLPredictions(unittest.TestCase):
    """Test cases for temporal ML prediction generation."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_patients = 800
        self.n_timesteps = 36

        # Create base risks and temporal risk matrix
        self.base_risks = assign_patient_risks(
            self.n_patients, annual_incident_rate=0.1,
            concentration=0.5, random_seed=42
        )

        self.temporal_matrix = build_temporal_risk_matrix(
            self.base_risks,
            n_timesteps=self.n_timesteps,
            random_seed=42
        )

    def test_generate_temporal_ml_predictions_basic(self):
        """Test basic temporal ML prediction generation."""
        preds, binary, metrics = generate_temporal_ml_predictions(
            self.temporal_matrix,
            prediction_start_time=10,
            prediction_window_length=12,
            random_seed=42
        )

        # Check output shapes
        self.assertEqual(preds.shape, (self.n_patients,))
        self.assertEqual(binary.shape, (self.n_patients,))
        self.assertIsInstance(metrics, dict)

        # Check predictions are valid probabilities
        self.assertTrue(np.all(preds >= 0))
        self.assertTrue(np.all(preds <= 1))

        # Check binary predictions are binary
        self.assertTrue(np.all(np.isin(binary, [0, 1])))

        # Check key metrics are present
        self.assertIn('sensitivity', metrics)
        self.assertIn('ppv', metrics)
        self.assertIn('temporal_correlation', metrics)
        self.assertIn('integrated_risk_correlation', metrics)

    def test_temporal_correlation_requirement(self):
        """Test that predictions correlate with temporal risk changes."""
        preds, binary, metrics = generate_temporal_ml_predictions(
            self.temporal_matrix,
            prediction_start_time=5,
            prediction_window_length=20,
            random_seed=42
        )

        # Check that temporal correlation exists and is reasonable
        if not np.isnan(metrics['temporal_correlation']):
            # Should have some correlation (positive or negative)
            self.assertGreater(abs(metrics['temporal_correlation']), 0.1)

        # Integrated risk correlation should be strong
        self.assertGreater(metrics['integrated_risk_correlation'], 0.3)

    def test_survival_integration_method(self):
        """Test survival integration method."""
        preds, binary, metrics = generate_temporal_ml_predictions(
            self.temporal_matrix,
            prediction_start_time=10,
            prediction_window_length=8,
            random_seed=42
        )

        # Check that method produces valid results
        self.assertEqual(len(preds), self.n_patients)
        self.assertIn('integration_method', metrics)
        self.assertEqual(metrics['integration_method'], 'survival')

        # Check that predictions are valid probabilities
        self.assertTrue(np.all(preds >= 0))
        self.assertTrue(np.all(preds <= 1))

        # Check that we have reasonable correlation with integrated risks
        self.assertIn('integrated_risk_correlation', metrics)
        self.assertGreater(metrics['integrated_risk_correlation'], 0.3)

    def test_performance_targets(self):
        """Test that performance targets are achieved within tolerance."""
        target_sens = 0.8
        target_ppv = 0.3
        tolerance = 0.05  # 5% tolerance

        preds, binary, metrics = generate_temporal_ml_predictions(
            self.temporal_matrix,
            prediction_start_time=15,
            prediction_window_length=10,
            target_sensitivity=target_sens,
            target_ppv=target_ppv,
            random_seed=42
        )

        # Check performance within tolerance
        sens_diff = abs(metrics['sensitivity'] - target_sens)
        ppv_diff = abs(metrics['ppv'] - target_ppv)

        self.assertLess(sens_diff, tolerance,
                        f"Sensitivity {metrics['sensitivity']:.3f} not within "
                        f"{tolerance:.1%} of target {target_sens:.3f}")
        self.assertLess(ppv_diff, tolerance,
                        f"PPV {metrics['ppv']:.3f} not within "
                        f"{tolerance:.1%} of target {target_ppv:.3f}")

    def test_window_bounds_validation(self):
        """Test validation of prediction window bounds."""
        # Test negative start time
        with self.assertRaises(ValueError):
            generate_temporal_ml_predictions(
                self.temporal_matrix,
                prediction_start_time=-1,
                prediction_window_length=10
            )

        # Test window extending beyond matrix
        with self.assertRaises(ValueError):
            generate_temporal_ml_predictions(
                self.temporal_matrix,
                prediction_start_time=45,
                prediction_window_length=15  # 45 + 15 = 60 > 52
            )

        # Test valid boundary case
        preds, binary, metrics = generate_temporal_ml_predictions(
            self.temporal_matrix,
            prediction_start_time=40,
            prediction_window_length=12,  # 40 + 12 = 52 (exactly at boundary)
            random_seed=42
        )

        self.assertEqual(len(preds), self.n_patients)

    def test_single_timestep_window(self):
        """Test edge case of single timestep prediction window."""
        preds, binary, metrics = generate_temporal_ml_predictions(
            self.temporal_matrix,
            prediction_start_time=25,
            prediction_window_length=1,
            random_seed=42
        )

        # Should work without errors
        self.assertEqual(len(preds), self.n_patients)

        # Temporal correlation should be NaN for single timestep
        self.assertTrue(np.isnan(metrics['temporal_correlation']))


class TestMLPredictionSimulatorTemporal(unittest.TestCase):
    """Test temporal methods of MLPredictionSimulator class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_patients = 600
        self.n_timesteps = 30

        self.base_risks = assign_patient_risks(
            self.n_patients, annual_incident_rate=0.1, random_seed=42
        )

        self.temporal_matrix = build_temporal_risk_matrix(
            self.base_risks,
            n_timesteps=self.n_timesteps,
            random_seed=42
        )

        self.simulator = MLPredictionSimulator(
            target_sensitivity=0.75,
            target_ppv=0.25,
            random_seed=42
        )

    def test_generate_temporal_predictions_method(self):
        """Test the temporal predictions method of MLPredictionSimulator."""
        preds, binary, info = self.simulator.generate_temporal_predictions(
            self.temporal_matrix,
            prediction_start_time=8,
            prediction_window_length=6
        )

        # Verify integration method is correctly set to survival
        self.assertEqual(info['integration_method'], 'survival')

        # Check outputs
        self.assertEqual(preds.shape, (self.n_patients,))
        self.assertEqual(binary.shape, (self.n_patients,))
        self.assertIsInstance(info, dict)

        # Check info content
        self.assertIn('integration_method', info)
        self.assertIn('window_start', info)
        self.assertIn('window_length', info)
        self.assertIn('temporal_correlation', info)
        self.assertIn('integration_correlation', info)

        # Verify values
        self.assertEqual(info['integration_method'], 'survival')
        self.assertEqual(info['window_start'], 8)
        self.assertEqual(info['window_length'], 6)

    def test_temporal_method_consistency(self):
        """Test temporal method produces valid and consistent structure."""
        # Run the same prediction twice
        preds1, binary1, info1 = self.simulator.generate_temporal_predictions(
            self.temporal_matrix,
            prediction_start_time=5,
            prediction_window_length=8
        )

        preds2, binary2, info2 = self.simulator.generate_temporal_predictions(
            self.temporal_matrix,
            prediction_start_time=5,
            prediction_window_length=8
        )

        # Both calls should produce valid outputs with same structure
        self.assertEqual(preds1.shape, preds2.shape)
        self.assertEqual(binary1.shape, binary2.shape)
        self.assertEqual(len(preds1), self.n_patients)
        self.assertEqual(len(preds2), self.n_patients)

        # Predictions should be valid probabilities
        self.assertTrue(np.all(preds1 >= 0) and np.all(preds1 <= 1))
        self.assertTrue(np.all(preds2 >= 0) and np.all(preds2 <= 1))

        # Info should be consistently structured
        self.assertEqual(info1['integration_method'], 'survival')
        self.assertEqual(info2['integration_method'], 'survival')
        self.assertEqual(info1['window_start'], info2['window_start'])
        self.assertEqual(info1['window_length'], info2['window_length'])

    def test_parameter_optimization_temporal(self):
        """Test that parameter optimization works with temporal data."""
        # First call should optimize parameters
        preds1, binary1, info1 = self.simulator.generate_temporal_predictions(
            self.temporal_matrix,
            prediction_start_time=5,
            prediction_window_length=10
        )

        # Check that parameters were set
        self.assertIsNotNone(self.simulator.noise_correlation)
        self.assertIsNotNone(self.simulator.noise_scale)

        # Second call should use existing parameters
        preds2, binary2, info2 = self.simulator.generate_temporal_predictions(
            self.temporal_matrix,
            prediction_start_time=10,
            prediction_window_length=8
        )

        # Should use same optimization parameters
        self.assertEqual(info1['optimization_correlation'],
                         info2['optimization_correlation'])
        self.assertEqual(info1['optimization_scale'],
                         info2['optimization_scale'])


@pytest.mark.skip(reason="Performance: Skip expensive temporal validation tests for CI")
class TestTemporalSensitivityValidation(unittest.TestCase):
    """Test temporal sensitivity validation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_patients = 600
        self.n_timesteps = 30

        base_risks = assign_patient_risks(
            self.n_patients, annual_incident_rate=0.12, random_seed=42
        )

        self.temporal_matrix = build_temporal_risk_matrix(
            base_risks,
            n_timesteps=self.n_timesteps,
            rho=0.85,  # High temporal correlation
            sigma=0.15,
            random_seed=42
        )

    def test_validate_temporal_sensitivity_basic(self):
        """Test basic temporal sensitivity validation."""
        # Generate predictions
        preds, _, _ = generate_temporal_ml_predictions(
            self.temporal_matrix,
            prediction_start_time=10,
            prediction_window_length=15,
            random_seed=42
        )

        # Validate temporal sensitivity
        validation = validate_temporal_sensitivity(
            self.temporal_matrix,
            preds,
            min_correlation=0.3
        )

        # Check validation structure
        expected_keys = [
            'mean_risk_correlation',
            'variance_correlation',
            'final_risk_correlation',
            'max_correlation',
            'mean_abs_correlation',
            'passes_threshold',
            'min_required_correlation'
        ]

        for key in expected_keys:
            self.assertIn(key, validation)

        # Check that correlation values are reasonable
        self.assertGreaterEqual(validation['max_correlation'], 0)
        self.assertLessEqual(validation['max_correlation'], 1)
        self.assertEqual(validation['min_required_correlation'], 0.3)

    def test_validation_thresholds(self):
        """Test validation with different correlation thresholds."""
        # Generate predictions with good temporal correlation
        preds, _, _ = generate_temporal_ml_predictions(
            self.temporal_matrix,
            prediction_start_time=5,
            prediction_window_length=20,
            random_seed=42
        )

        # Test with low threshold (should pass)
        validation_low = validate_temporal_sensitivity(
            self.temporal_matrix, preds, min_correlation=0.1
        )
        self.assertTrue(validation_low['passes_threshold'])

        # Test with very high threshold (might fail)
        validation_high = validate_temporal_sensitivity(
            self.temporal_matrix, preds, min_correlation=0.95
        )
        # Don't assert pass/fail as it depends on actual correlation achieved
        self.assertIsInstance(validation_high['passes_threshold'], bool)

    def test_constant_predictions_edge_case(self):
        """Test validation with constant predictions (edge case)."""
        # Create constant predictions
        constant_preds = np.full(self.n_patients, 0.5)

        validation = validate_temporal_sensitivity(
            self.temporal_matrix,
            constant_preds,
            min_correlation=0.1
        )

        # Should handle constant predictions gracefully
        self.assertEqual(validation['mean_risk_correlation'], 0.0)
        self.assertEqual(validation['variance_correlation'], 0.0)
        self.assertEqual(validation['final_risk_correlation'], 0.0)
        self.assertFalse(validation['passes_threshold'])


@pytest.mark.skip(reason="Performance: Skip expensive benchmarking tests for CI")
class TestTemporalMLBenchmarking(unittest.TestCase):
    """Test temporal ML performance benchmarking functionality."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_patients = 600
        self.n_timesteps = 30

        self.base_risks = assign_patient_risks(
            self.n_patients, annual_incident_rate=0.08, random_seed=42
        )

        self.temporal_matrix = build_temporal_risk_matrix(
            self.base_risks,
            n_timesteps=self.n_timesteps,
            random_seed=42
        )

    def test_benchmark_temporal_ml_performance_basic(self):
        """Test basic benchmarking functionality."""
        configs = [
            {
                'start_time': 5,
                'window_length': 8
            },
            {
                'start_time': 10,
                'window_length': 6
            },
            {
                'start_time': 15,
                'window_length': 10
            }
        ]

        results_df = benchmark_temporal_ml_performance(
            self.temporal_matrix,
            self.base_risks,
            configs,
            random_seed=42
        )

        # Check output structure
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertEqual(len(results_df), len(configs))

        # Check expected columns exist
        expected_cols = [
            'config_id', 'start_time', 'window_length', 'integration_method',
            'temporal_sensitivity', 'temporal_ppv', 'temporal_f1',
            'static_sensitivity', 'static_ppv', 'static_f1'
        ]

        for col in expected_cols:
            self.assertIn(col, results_df.columns)

        # Check configuration values match
        for i, config in enumerate(configs):
            row = results_df.iloc[i]
            self.assertEqual(row['start_time'], config['start_time'])
            self.assertEqual(row['window_length'], config['window_length'])
            # Integration method should always be 'survival' now
            self.assertEqual(row['integration_method'], 'survival')

    def test_benchmark_comparison_temporal_vs_static(self):
        """Test that benchmark compares temporal vs static approaches."""
        configs = [
            {
                'start_time': 8,
                'window_length': 12
            }
        ]

        results_df = benchmark_temporal_ml_performance(
            self.temporal_matrix,
            self.base_risks,
            configs,
            random_seed=42
        )

        # Verify integration method is always set to survival
        row = results_df.iloc[0]
        self.assertEqual(row['integration_method'], 'survival')

        # Both temporal and static should have reasonable performance
        self.assertGreater(row['temporal_sensitivity'], 0.1)
        self.assertLessEqual(row['temporal_sensitivity'], 1.0)
        self.assertGreater(row['static_sensitivity'], 0.1)
        self.assertLessEqual(row['static_sensitivity'], 1.0)

        # PPV values should be reasonable
        self.assertGreater(row['temporal_ppv'], 0.05)
        self.assertLess(row['temporal_ppv'], 0.8)
        self.assertGreater(row['static_ppv'], 0.05)
        self.assertLess(row['static_ppv'], 0.8)

    def test_benchmark_error_handling(self):
        """Test benchmarking with invalid configurations."""
        configs = [
            # Valid config
            {
                'start_time': 5,
                'window_length': 8
            },
            # Invalid config (window extends beyond matrix)
            {
                'start_time': 25,
                'window_length': 20  # 25 + 20 = 45 > 30
            }
        ]

        results_df = benchmark_temporal_ml_performance(
            self.temporal_matrix,
            self.base_risks,
            configs,
            random_seed=42
        )

        # Should still return results for all configs
        self.assertEqual(len(results_df), 2)

        # First config should succeed
        self.assertNotIn('error', results_df.iloc[0])

        # Second config should have error
        self.assertIn('error', results_df.iloc[1])


@pytest.mark.skip(reason="Performance: Skip expensive integration tests for CI")
class TestIntegrationWithExistingWorkflow(unittest.TestCase):
    """Test integration with existing ML simulation workflow."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_patients = 500
        self.n_timesteps = 24

        self.base_risks = assign_patient_risks(
            self.n_patients, annual_incident_rate=0.15, random_seed=42
        )

        self.temporal_matrix = build_temporal_risk_matrix(
            self.base_risks,
            n_timesteps=self.n_timesteps,
            random_seed=42
        )

    def test_compatibility_with_existing_functions(self):
        """Test that temporal predictions work with existing evaluation."""
        from pop_ml_simulator.ml_simulation import (
            evaluate_threshold_based,
            evaluate_topk,
            hosmer_lemeshow_test
        )

        # Generate temporal predictions
        preds, binary, metrics = generate_temporal_ml_predictions(
            self.temporal_matrix,
            prediction_start_time=8,
            prediction_window_length=10,
            random_seed=42
        )

        # Create mock true labels for evaluation
        np.random.seed(43)
        true_labels = np.random.binomial(1, preds * 0.8)

        # Test threshold-based evaluation
        threshold_metrics = evaluate_threshold_based(
            true_labels, preds, threshold=0.3
        )
        self.assertIn('sensitivity', threshold_metrics)
        self.assertIn('ppv', threshold_metrics)

        # Test TopK evaluation
        topk_metrics = evaluate_topk(true_labels, preds, k_percent=20)
        self.assertIn('sensitivity', topk_metrics)
        self.assertIn('ppv', topk_metrics)

        # Test Hosmer-Lemeshow calibration
        if len(np.unique(true_labels)) > 1:  # Only if we have both classes
            hl_stat, p_value = hosmer_lemeshow_test(true_labels, preds)
            self.assertIsInstance(hl_stat, float)
            self.assertIsInstance(p_value, float)

    def test_temporal_vs_static_performance_difference(self):
        """Test that temporal and static approaches yield different results."""
        # Generate temporal predictions
        temporal_preds, _, temporal_metrics = generate_temporal_ml_predictions(
            self.temporal_matrix,
            prediction_start_time=5,
            prediction_window_length=15,
            random_seed=42
        )

        # Generate static predictions using base risks
        static_simulator = MLPredictionSimulator(
            target_sensitivity=0.8,
            target_ppv=0.3,
            random_seed=42
        )

        # Create labels for static approach
        np.random.seed(44)
        static_labels = np.random.binomial(1, self.base_risks * 1.2)

        static_params = static_simulator.optimize_noise_parameters(
            static_labels, self.base_risks, n_iterations=10
        )

        static_preds, _ = static_simulator.generate_predictions(
            static_labels,
            self.base_risks,
            static_params['correlation'],
            static_params['scale']
        )

        # Predictions should be different
        correlation = np.corrcoef(temporal_preds, static_preds)[0, 1]
        self.assertLess(correlation, 0.95,
                        "Temporal and static predictions too similar")

        # Both should be valid probabilities
        self.assertTrue(np.all(temporal_preds >= 0))
        self.assertTrue(np.all(temporal_preds <= 1))
        self.assertTrue(np.all(static_preds >= 0))
        self.assertTrue(np.all(static_preds <= 1))


if __name__ == '__main__':
    unittest.main()
