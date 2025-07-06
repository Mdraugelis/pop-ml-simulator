"""
Tests for risk integration module.
"""

import unittest
import numpy as np
import warnings

from pop_ml_simulator.risk_integration import (
    integrate_window_risk,
    extract_risk_windows,
    validate_integration_bounds
)


class TestSurvivalRiskIntegration(unittest.TestCase):
    """Test cases for survival-based window risk integration."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_patients = 100
        self.window_length = 12

        # Create test risk windows
        self.constant_risks = np.full(
            (self.n_patients, self.window_length), 0.1
        )
        self.increasing_risks = np.linspace(0.05, 0.2, self.window_length)
        self.increasing_risks = np.tile(
            self.increasing_risks, (self.n_patients, 1)
        )
        self.random_risks = np.random.uniform(
            0.05, 0.15, (self.n_patients, self.window_length)
        )

    def test_survival_integration_basic(self):
        """Test survival-based integration method."""
        # Single patient constant risk
        single_risk = np.array([0.1, 0.1, 0.1, 0.1])
        integrated = integrate_window_risk(single_risk)

        # For constant risk, integrated should be higher than single period
        self.assertGreater(integrated, 0.1)
        self.assertLess(integrated, 0.4)  # But less than sum
        self.assertAlmostEqual(integrated, 0.344, places=3)

        # Multiple patients
        integrated_multi = integrate_window_risk(self.constant_risks)
        self.assertEqual(integrated_multi.shape, (self.n_patients,))
        self.assertTrue(np.all(integrated_multi > 0.1))
        self.assertTrue(np.all(integrated_multi < 1.0))

    def test_average_integration(self):
        """Test simple averaging integration method."""
        # Constant risks should return same value
        integrated = integrate_window_risk(
            self.constant_risks, integration_method='average'
        )
        np.testing.assert_array_almost_equal(integrated, 0.1)

        # Increasing risks
        integrated_inc = integrate_window_risk(
            self.increasing_risks, integration_method='average'
        )
        expected_avg = np.mean(self.increasing_risks[0])
        np.testing.assert_array_almost_equal(integrated_inc, expected_avg)

    def test_weighted_recent_integration(self):
        """Test weighted recent integration method."""
        # For increasing risks, weighted should be higher than average
        integrated_weighted = integrate_window_risk(
            self.increasing_risks, integration_method='weighted_recent'
        )
        integrated_avg = integrate_window_risk(
            self.increasing_risks, integration_method='average'
        )

        # Weighted should favor recent (higher) values
        self.assertTrue(np.all(integrated_weighted > integrated_avg))

        # Should still be valid probabilities
        self.assertTrue(np.all(integrated_weighted >= 0))
        self.assertTrue(np.all(integrated_weighted <= 1))

    def test_integration_monotonicity(self):
        """Test that increasing risks yield higher integrated values."""
        # Create two risk trajectories
        low_risks = np.full((1, self.window_length), 0.05)
        high_risks = np.full((1, self.window_length), 0.15)

        for method in ['survival', 'average', 'weighted_recent']:
            integrated_low = integrate_window_risk(
                low_risks, integration_method=method
            )
            integrated_high = integrate_window_risk(
                high_risks, integration_method=method
            )

            self.assertLess(
                integrated_low, integrated_high,
                f"Method {method} failed monotonicity test"
            )

    def test_integration_edge_cases(self):
        """Test edge cases: all zeros, all ones, single timestep."""
        # All zeros
        zero_risks = np.zeros((self.n_patients, self.window_length))
        for method in ['survival', 'average', 'weighted_recent']:
            integrated = integrate_window_risk(
                zero_risks, integration_method=method
            )
            np.testing.assert_array_almost_equal(
                integrated, 0.0,
                err_msg=f"Method {method} failed for all zeros"
            )

        # All ones (should handle without errors)
        one_risks = np.ones((self.n_patients, self.window_length))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Expect warnings for risk=1
            for method in ['survival', 'average', 'weighted_recent']:
                integrated = integrate_window_risk(
                    one_risks, integration_method=method
                )
                self.assertTrue(np.all(integrated <= 1.0))
                self.assertTrue(np.all(integrated >= 0.99))

        # Single timestep
        single_step = np.random.uniform(
            0.05, 0.15, (self.n_patients, 1)
        )
        for method in ['survival', 'average', 'weighted_recent']:
            integrated = integrate_window_risk(
                single_step, integration_method=method
            )
            # For single timestep, all methods should return same value
            np.testing.assert_array_almost_equal(
                integrated, single_step.squeeze(),
                err_msg=f"Method {method} failed for single timestep"
            )

    def test_survival_integration_properties(self):
        """Test mathematical properties of survival integration."""
        # Property 1: For very small risks, should approximate sum
        small_risks = np.full((1, 4), 0.01)
        integrated = integrate_window_risk(
            small_risks, integration_method='survival'
        )
        sum_risks = np.sum(small_risks)
        self.assertAlmostEqual(integrated[0], sum_risks, places=2)

        # Property 2: Should never exceed 1.0
        large_risks = np.full((1, 52), 0.5)  # 50% weekly risk for a year
        integrated = integrate_window_risk(large_risks)
        self.assertLessEqual(integrated[0], 1.0)
        self.assertGreater(integrated[0], 0.99)  # Should be very close to 1

        # Property 3: Order matters for varying risks
        # Compare [0.1, 0.2] vs [0.2, 0.1] - should give same result
        risks_a = np.array([[0.1, 0.2]])
        risks_b = np.array([[0.2, 0.1]])
        integrated_a = integrate_window_risk(
            risks_a, integration_method='survival'
        )
        integrated_b = integrate_window_risk(
            risks_b, integration_method='survival'
        )
        # For survival method, order doesn't matter (cumulative hazard)
        self.assertAlmostEqual(integrated_a[0], integrated_b[0], places=5)

    def test_different_timestep_durations(self):
        """Test integration with different timestep durations."""
        risks = np.full((1, 12), 0.01)  # 1% risk per timestep

        # Weekly timesteps (default)
        weekly = integrate_window_risk(risks, timestep_duration=1/52)

        # Monthly timesteps
        monthly = integrate_window_risk(risks, timestep_duration=1/12)

        # Daily timesteps
        daily = integrate_window_risk(risks, timestep_duration=1/365)

        # For survival method, all should give same result
        # because we're converting back and forth
        self.assertAlmostEqual(monthly[0], weekly[0], places=5)
        self.assertAlmostEqual(weekly[0], daily[0], places=5)

    def test_input_validation(self):
        """Test input validation and error handling."""
        # Invalid method
        with self.assertRaises(ValueError):
            integrate_window_risk(
                self.random_risks, integration_method='invalid'
            )

        # Invalid shape
        with self.assertRaises(ValueError):
            integrate_window_risk(np.zeros((10, 10, 10)))

        # Risks outside [0, 1] should warn and clip
        invalid_risks = np.array([[-0.1, 0.5, 1.2]])
        with warnings.catch_warnings(record=True) as w:
            integrated = integrate_window_risk(invalid_risks)
            self.assertTrue(len(w) > 0)
            self.assertTrue(0 <= integrated[0] <= 1)

    def test_vectorization_performance(self):
        """Test that methods are efficiently vectorized."""
        # Large matrix
        large_risks = np.random.uniform(0.05, 0.15, size=(10000, 52))

        import time
        start = time.time()
        integrated = integrate_window_risk(large_risks)
        elapsed = time.time() - start

        # Should process 10k patients in under 1 second
        self.assertLess(elapsed, 1.0)
        self.assertEqual(integrated.shape, (10000,))


class TestRiskWindowExtraction(unittest.TestCase):
    """Test cases for risk window extraction utilities."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_patients = 100
        self.total_timesteps = 52
        self.temporal_risks = np.random.uniform(
            0.05, 0.15,
            size=(self.n_patients, self.total_timesteps)
        )

    def test_extract_windows_basic(self):
        """Test basic window extraction."""
        # Extract 12-week window starting at week 10
        windows = extract_risk_windows(
            self.temporal_risks,
            start_time=10,
            window_length=12
        )

        self.assertEqual(windows.shape, (self.n_patients, 12))
        np.testing.assert_array_equal(
            windows,
            self.temporal_risks[:, 10:22]
        )

    def test_extract_windows_edge_cases(self):
        """Test edge cases for window extraction."""
        # Start at beginning
        windows = extract_risk_windows(
            self.temporal_risks,
            start_time=0,
            window_length=10
        )
        self.assertEqual(windows.shape, (self.n_patients, 10))

        # End at last timestep
        windows = extract_risk_windows(
            self.temporal_risks,
            start_time=40,
            window_length=12
        )
        self.assertEqual(windows.shape, (self.n_patients, 12))

        # Single timestep window
        windows = extract_risk_windows(
            self.temporal_risks,
            start_time=25,
            window_length=1
        )
        self.assertEqual(windows.shape, (self.n_patients, 1))

    def test_extract_windows_errors(self):
        """Test error handling for invalid window extraction."""
        # Negative start time
        with self.assertRaises(ValueError):
            extract_risk_windows(
                self.temporal_risks,
                start_time=-1,
                window_length=10
            )

        # Window extends beyond available data
        with self.assertRaises(ValueError):
            extract_risk_windows(
                self.temporal_risks,
                start_time=45,
                window_length=10
            )

        # Start time beyond data
        with self.assertRaises(ValueError):
            extract_risk_windows(
                self.temporal_risks,
                start_time=60,
                window_length=5
            )


class TestIntegrationBoundsValidation(unittest.TestCase):
    """Test cases for integration bounds validation."""

    def test_valid_bounds(self):
        """Test validation of valid probability bounds."""
        valid_risks = np.array([0.0, 0.1, 0.5, 0.99, 1.0])
        self.assertTrue(validate_integration_bounds(valid_risks))

    def test_invalid_bounds(self):
        """Test detection of invalid bounds."""
        # Negative values
        with warnings.catch_warnings(record=True) as w:
            negative_risks = np.array([-0.1, 0.5, 0.8])
            self.assertFalse(validate_integration_bounds(negative_risks))
            self.assertTrue(len(w) > 0)

        # Values > 1
        with warnings.catch_warnings(record=True) as w:
            large_risks = np.array([0.5, 1.1, 0.8])
            self.assertFalse(validate_integration_bounds(large_risks))
            self.assertTrue(len(w) > 0)

    def test_numerical_tolerance(self):
        """Test numerical tolerance in bounds checking."""
        # Values very close to bounds should pass
        edge_risks = np.array([-1e-12, 1 + 1e-12])
        self.assertTrue(
            validate_integration_bounds(edge_risks, tolerance=1e-10)
        )

        # But not if outside tolerance
        self.assertFalse(
            validate_integration_bounds(edge_risks, tolerance=1e-15)
        )


class TestSurvivalIntegrationProperties(unittest.TestCase):
    """Test specific properties of survival-based integration."""

    def test_constant_risk_properties(self):
        """Test survival integration for constant risks."""
        constant_risk = np.full((100, 12), 0.1)
        survival = integrate_window_risk(constant_risk)
        
        # Should be higher than single timestep due to cumulative nature
        self.assertTrue(np.all(survival > 0.1))
        # But should be valid probabilities
        self.assertTrue(np.all(survival <= 1.0))

    def test_risk_ordering_preservation(self):
        """Test that survival integration preserves risk ordering."""
        # Create patients with different risk levels
        n_patients = 5
        risk_levels = [0.05, 0.1, 0.15, 0.2, 0.25]
        risks = np.array([np.full(12, level) for level in risk_levels])

        integrated = integrate_window_risk(risks)

        # Check that ordering is preserved
        for i in range(n_patients - 1):
            self.assertLess(
                integrated[i], integrated[i + 1],
                "Survival integration failed to preserve risk ordering"
            )


if __name__ == '__main__':
    unittest.main()
