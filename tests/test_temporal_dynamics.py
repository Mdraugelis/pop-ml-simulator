"""
Tests for temporal dynamics module.
"""

import unittest
import numpy as np
import pytest
from pop_ml_simulator.temporal_dynamics import (
    simulate_ar1_process,
    TemporalRiskSimulator,
    EnhancedTemporalRiskSimulator,
    build_temporal_risk_matrix
)


class TestAR1Process(unittest.TestCase):
    """Test cases for AR(1) process simulation."""

    @pytest.mark.skip(reason="Performance: Skip expensive AR1 simulation")
    def test_simulate_ar1_basic(self):
        """Test basic AR(1) simulation."""
        n_timesteps = 50
        rho = 0.9
        sigma = 0.1

        values = simulate_ar1_process(n_timesteps, rho, sigma, random_seed=42)

        # Check output shape
        self.assertEqual(len(values), n_timesteps)

        # Check all values are within bounds
        self.assertTrue(np.all(values >= 0.5))
        self.assertTrue(np.all(values <= 2.0))

        # Check mean is approximately 1.0
        self.assertAlmostEqual(np.mean(values), 1.0, delta=0.1)

    def test_simulate_ar1_persistence(self):
        """Test that persistence parameter affects autocorrelation."""
        n_timesteps = 200
        sigma = 0.1

        # High persistence
        values_high = simulate_ar1_process(
            n_timesteps, rho=0.95, sigma=sigma, random_seed=42)
        # Low persistence
        values_low = simulate_ar1_process(
            n_timesteps, rho=0.5, sigma=sigma, random_seed=42)

        # Calculate lag-1 autocorrelation
        def autocorr(x):
            return np.corrcoef(x[:-1], x[1:])[0, 1]

        self.assertGreater(autocorr(values_high), autocorr(values_low))

        # Should be close to theoretical values
        self.assertAlmostEqual(autocorr(values_high), 0.95, delta=0.05)
        self.assertAlmostEqual(autocorr(values_low), 0.5, delta=0.1)

    def test_simulate_ar1_custom_bounds(self):
        """Test with custom bounds."""
        values = simulate_ar1_process(
            100, 0.9, 0.1, bounds=(0.8, 1.2), random_seed=42)

        self.assertTrue(np.all(values >= 0.8))
        self.assertTrue(np.all(values <= 1.2))

    def test_simulate_ar1_initial_value(self):
        """Test with custom initial value."""
        initial = 1.5
        values = simulate_ar1_process(
            10, 0.9, 0.1, initial_value=initial, random_seed=42)

        self.assertEqual(values[0], initial)

    def test_simulate_ar1_reproducibility(self):
        """Test reproducibility with random seed."""
        values1 = simulate_ar1_process(50, 0.9, 0.1, random_seed=42)
        values2 = simulate_ar1_process(50, 0.9, 0.1, random_seed=42)

        np.testing.assert_array_equal(values1, values2)


class TestTemporalRiskSimulator(unittest.TestCase):
    """Test cases for TemporalRiskSimulator class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_patients = 100
        self.base_risks = np.random.uniform(0.05, 0.2, self.n_patients)

    def test_initialization(self):
        """Test simulator initialization."""
        sim = TemporalRiskSimulator(self.base_risks, rho=0.9, sigma=0.1)

        self.assertEqual(sim.n_patients, self.n_patients)
        self.assertEqual(sim.rho, 0.9)
        self.assertEqual(sim.sigma, 0.1)
        self.assertEqual(sim.bounds, (0.5, 2.0))

        # Check initial state
        np.testing.assert_array_equal(
            sim.current_modifiers, np.ones(self.n_patients))
        self.assertEqual(len(sim.modifier_history), 1)
        self.assertEqual(len(sim.risk_history), 1)

    def test_step(self):
        """Test single step update."""
        sim = TemporalRiskSimulator(self.base_risks, rho=0.9, sigma=0.1)

        # Take one step
        sim.step()

        # Should have 2 entries in history
        self.assertEqual(len(sim.modifier_history), 2)
        self.assertEqual(len(sim.risk_history), 2)

        # Modifiers should have changed but still be within bounds
        self.assertFalse(np.array_equal(
            sim.current_modifiers, np.ones(self.n_patients)))
        self.assertTrue(np.all(sim.current_modifiers >= 0.5))
        self.assertTrue(np.all(sim.current_modifiers <= 2.0))

    def test_simulate(self):
        """Test multi-step simulation."""
        sim = TemporalRiskSimulator(self.base_risks, rho=0.9, sigma=0.1)

        n_steps = 26
        sim.simulate(n_steps)

        # Should have n_steps + 1 entries (including initial)
        self.assertEqual(len(sim.modifier_history), n_steps + 1)
        self.assertEqual(len(sim.risk_history), n_steps + 1)

    def test_get_current_risks(self):
        """Test current risk calculation."""
        sim = TemporalRiskSimulator(self.base_risks, rho=0.9, sigma=0.1)

        # Initially should equal base risks
        initial_risks = sim.get_current_risks()
        np.testing.assert_array_almost_equal(initial_risks, self.base_risks)

        # After steps, should be modified
        sim.simulate(10)
        current_risks = sim.get_current_risks()

        # Should be base_risks * modifiers
        expected = self.base_risks * sim.current_modifiers
        np.testing.assert_array_almost_equal(current_risks, expected)

    def test_get_histories(self):
        """Test history retrieval."""
        sim = TemporalRiskSimulator(self.base_risks, rho=0.9, sigma=0.1)
        sim.simulate(10)

        modifier_hist, risk_hist = sim.get_histories()

        # Check shapes - should be (n_patients, n_timesteps)
        self.assertEqual(modifier_hist.shape, (self.n_patients, 11))
        self.assertEqual(risk_hist.shape, (self.n_patients, 11))

        # Check first timestep
        np.testing.assert_array_almost_equal(
            modifier_hist[:, 0], np.ones(self.n_patients))
        np.testing.assert_array_almost_equal(risk_hist[:, 0], self.base_risks)

    def test_population_mean_preservation(self):
        """Test that population mean risk is preserved over time."""
        sim = TemporalRiskSimulator(self.base_risks, rho=0.9, sigma=0.1)
        sim.simulate(50)

        _, risk_hist = sim.get_histories()

        # Population mean should stay close to initial mean
        initial_mean = np.mean(self.base_risks)
        final_mean = np.mean(risk_hist[:, -1])

        self.assertAlmostEqual(final_mean, initial_mean, delta=0.01)


class TestEnhancedTemporalRiskSimulator(unittest.TestCase):
    """Test cases for EnhancedTemporalRiskSimulator class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_patients = 100
        self.base_risks = np.random.uniform(0.05, 0.2, self.n_patients)

    def test_initialization(self):
        """Test enhanced simulator initialization."""
        sim = EnhancedTemporalRiskSimulator(
            self.base_risks,
            seasonal_amplitude=0.2,
            seasonal_period=52
        )

        self.assertEqual(sim.seasonal_amplitude, 0.2)
        self.assertEqual(sim.seasonal_period, 52)
        self.assertEqual(sim.time_step, 0)
        self.assertEqual(len(sim.external_shocks), 0)

    def test_seasonal_modifier(self):
        """Test seasonal modifier calculation."""
        sim = EnhancedTemporalRiskSimulator(
            self.base_risks,
            seasonal_amplitude=0.2,
            seasonal_period=52
        )

        # At time 0 (winter peak)
        winter_mod = sim.get_seasonal_modifier()
        self.assertAlmostEqual(winter_mod, 1.2, places=2)

        # At time 26 (summer trough)
        sim.time_step = 26
        summer_mod = sim.get_seasonal_modifier()
        self.assertAlmostEqual(summer_mod, 0.8, places=2)

    def test_add_shock(self):
        """Test adding external shocks."""
        sim = EnhancedTemporalRiskSimulator(self.base_risks)

        sim.add_shock(
            time_step=10,
            magnitude=1.5,
            duration=4,
            affected_fraction=0.5,
            random_seed=42
        )

        self.assertEqual(len(sim.external_shocks), 1)
        shock = sim.external_shocks[0]
        self.assertEqual(shock['time'], 10)
        self.assertEqual(shock['magnitude'], 1.5)
        self.assertEqual(shock['duration'], 4)
        self.assertEqual(len(shock['affected_patients']), 50)

    def test_shock_modifier(self):
        """Test shock modifier calculation."""
        sim = EnhancedTemporalRiskSimulator(self.base_risks)

        # Add a shock
        sim.add_shock(
            time_step=5, magnitude=2.0, duration=3, affected_fraction=0.5)

        # Before shock
        sim.time_step = 4
        mod_before = sim.get_shock_modifier()
        np.testing.assert_array_equal(mod_before, np.ones(self.n_patients))

        # During shock
        sim.time_step = 6
        mod_during = sim.get_shock_modifier()
        affected = sim.external_shocks[0]['affected_patients']
        self.assertTrue(np.all(mod_during[affected] == 2.0))

        # After shock
        sim.time_step = 10
        mod_after = sim.get_shock_modifier()
        np.testing.assert_array_equal(mod_after, np.ones(self.n_patients))

    def test_step_with_seasonal_and_shock(self):
        """Test step update with both seasonal and shock effects."""
        sim = EnhancedTemporalRiskSimulator(
            self.base_risks,
            seasonal_amplitude=0.2,
            seasonal_period=52
        )

        # Add a shock
        sim.add_shock(
            time_step=1, magnitude=1.5, duration=2, affected_fraction=0.5)

        # Take a step during the shock
        sim.step()

        # Check that modifiers are applied
        self.assertEqual(sim.time_step, 1)

        # Modifiers should reflect both seasonal and shock effects
        # (seasonal and shock modifiers are calculated internally)

        # Some patients should have higher modifiers due to shock
        affected = sim.external_shocks[0]['affected_patients']
        self.assertTrue(np.any(
            sim.current_modifiers[affected] > sim.current_modifiers[~affected]
        ))

    def test_seasonal_effects_on_population_risk(self):
        """Test that seasonal effects properly modulate population risk."""
        # Create simulator with strong seasonal effects
        sim = EnhancedTemporalRiskSimulator(
            self.base_risks,
            seasonal_amplitude=0.3,  # 30% seasonal variation
            seasonal_period=52,
            rho=0.95,  # High persistence to isolate seasonal effect
            sigma=0.01  # Low noise to see clear pattern
        )

        # Track population risks over full seasonal cycle
        population_risks = []
        seasonal_modifiers = []

        for _ in range(52):
            sim.step()
            risks = sim.get_current_risks()
            population_risks.append(np.mean(risks))
            seasonal_modifiers.append(sim.get_seasonal_modifier())

        # Verify seasonal pattern exists
        risk_array = np.array(population_risks)
        modifier_array = np.array(seasonal_modifiers)

        # Check that population risk varies
        risk_variation = np.max(risk_array) - np.min(risk_array)
        self.assertGreater(risk_variation, 0.01,
                           "Population risk should show seasonal variation")

        # Check correlation between seasonal modifier and population risk
        # They should be positively correlated (but AR(1) process adds noise)
        correlation = np.corrcoef(modifier_array[1:], risk_array[1:])[0, 1]
        self.assertGreater(
            correlation, 0.2,
            f"Seasonal modifier and population risk should be "
            f"correlated, got {correlation}")

        # Check that risk shows clear seasonal pattern
        # The AR(1) process can introduce lag, so we check for overall pattern
        # rather than exact peak alignment
        risk_fft = np.fft.fft(risk_array - np.mean(risk_array))
        power_spectrum = np.abs(risk_fft) ** 2

        # The seasonal frequency should have significant power
        # (excluding DC component at index 0)
        seasonal_freq_idx = 1  # For annual cycle in 52 weeks
        self.assertGreater(
            power_spectrum[seasonal_freq_idx],
            np.median(power_spectrum[2:26]),  # Compare to other frequencies
            "Seasonal frequency should have significant power")

        # Verify average stays close to target
        avg_risk = np.mean(risk_array)
        target_risk = np.mean(self.base_risks)
        self.assertAlmostEqual(
            avg_risk, target_risk, delta=0.05,
            msg=f"Average risk {avg_risk:.4f} should be close to "
                f"target {target_risk:.4f}")


class TestTemporalRiskMatrix(unittest.TestCase):
    """Test cases for temporal risk matrix functionality."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_patients = 100
        self.n_timesteps = 12
        self.base_risks = np.random.uniform(0.05, 0.2, self.n_patients)

    def test_temporal_risk_matrix_construction(self):
        """Test basic temporal risk matrix construction."""
        risk_matrix = build_temporal_risk_matrix(
            self.base_risks,
            n_timesteps=self.n_timesteps,
            random_seed=42
        )

        # Check shape
        self.assertEqual(risk_matrix.shape,
                         (self.n_patients, self.n_timesteps))

        # Check that matrix is not empty
        self.assertGreater(np.count_nonzero(risk_matrix), 0)

        # Check that all values are finite
        self.assertTrue(np.all(np.isfinite(risk_matrix)))

    def test_temporal_risk_bounds(self):
        """Test that all risks remain in [0,1] range."""
        risk_matrix = build_temporal_risk_matrix(
            self.base_risks,
            n_timesteps=self.n_timesteps,
            max_risk_threshold=0.95,
            random_seed=42
        )

        # Check bounds
        self.assertTrue(np.all(risk_matrix >= 0.0),
                        f"Found negative risks: min={np.min(risk_matrix)}")
        self.assertTrue(np.all(risk_matrix <= 1.0),
                        f"Found risks > 1.0: max={np.max(risk_matrix)}")

        # Check that no risk exceeds the threshold
        self.assertTrue(np.all(risk_matrix <= 0.95),
                        f"Found risks > threshold: max={np.max(risk_matrix)}")

    def test_temporal_risk_initial_conditions(self):
        """Test that initial timestep matches base risks exactly."""
        risk_matrix = build_temporal_risk_matrix(
            self.base_risks,
            n_timesteps=self.n_timesteps,
            random_seed=42
        )

        # Check initial conditions
        np.testing.assert_array_almost_equal(
            risk_matrix[:, 0], self.base_risks,
            decimal=10,
            err_msg="Initial timestep should match base risks exactly"
        )

    @pytest.mark.skip(reason="Performance: Skip autocorrelation test")
    def test_temporal_autocorrelation(self):
        """Test temporal autocorrelation > 0.8 for patient trajectories."""
        risk_matrix = build_temporal_risk_matrix(
            self.base_risks,
            n_timesteps=26,  # Need fewer timesteps for performance
            rho=0.9,
            random_seed=42
        )

        autocorrelations = []
        # Sample of patients
        for patient_idx in range(min(20, self.n_patients)):
            trajectory = risk_matrix[patient_idx, :]
            if len(trajectory) > 1:
                # Calculate lag-1 autocorrelation
                corr = np.corrcoef(trajectory[:-1], trajectory[1:])[0, 1]
                if not np.isnan(corr):
                    autocorrelations.append(corr)

        # Check that average autocorrelation is > 0.8
        avg_autocorr = np.mean(autocorrelations)
        self.assertGreater(
            avg_autocorr, 0.8,
            f"Average autocorrelation {avg_autocorr:.3f} should be > 0.8")

    def test_matrix_access_methods(self):
        """Test the matrix access methods on EnhancedTemporalRiskSimulator."""
        np.random.seed(42)  # Set seed before creating simulator
        simulator = EnhancedTemporalRiskSimulator(
            self.base_risks,
            rho=0.9,
            sigma=0.1
        )

        # Run simulation
        simulator.simulate(self.n_timesteps - 1)

        # Test get_risk_matrix
        risk_matrix = simulator.get_risk_matrix()
        self.assertEqual(risk_matrix.shape,
                         (self.n_patients, self.n_timesteps))

        # Test get_patient_trajectory
        patient_id = 5
        trajectory = simulator.get_patient_trajectory(patient_id)
        self.assertEqual(len(trajectory), self.n_timesteps)
        np.testing.assert_array_equal(trajectory, risk_matrix[patient_id, :])

        # Test get_timestep_risks
        timestep = 10
        timestep_risks = simulator.get_timestep_risks(timestep)
        self.assertEqual(len(timestep_risks), self.n_patients)
        np.testing.assert_array_equal(timestep_risks, risk_matrix[:, timestep])

    def test_matrix_access_error_handling(self):
        """Test error handling in matrix access methods."""
        simulator = EnhancedTemporalRiskSimulator(self.base_risks)

        # Note: The simulator starts with initial state, so these should work
        # Just verify they don't raise errors for valid initial state
        risk_matrix = simulator.get_risk_matrix()
        self.assertEqual(risk_matrix.shape[0], self.n_patients)
        self.assertEqual(risk_matrix.shape[1], 1)  # Initial state only

        trajectory = simulator.get_patient_trajectory(0)
        self.assertEqual(len(trajectory), 1)

        timestep_risks = simulator.get_timestep_risks(0)
        self.assertEqual(len(timestep_risks), self.n_patients)

        # Run simulation
        simulator.simulate(self.n_timesteps - 1)

        # Test out-of-bounds patient ID
        with self.assertRaises(ValueError):
            simulator.get_patient_trajectory(-1)

        with self.assertRaises(ValueError):
            simulator.get_patient_trajectory(self.n_patients)

        # Test out-of-bounds timestep
        with self.assertRaises(ValueError):
            simulator.get_timestep_risks(-1)

        with self.assertRaises(ValueError):
            simulator.get_timestep_risks(self.n_timesteps)

    @pytest.mark.skip(reason="Performance: Skip expensive performance")
    def test_temporal_risk_matrix_performance(self):
        """Test performance for large matrices."""
        import time

        # Test with smaller matrix for CI (1000x104 might be too slow for CI)
        large_base_risks = np.random.uniform(0.05, 0.2, 200)
        large_timesteps = 26

        start_time = time.time()
        risk_matrix = build_temporal_risk_matrix(
            large_base_risks,
            n_timesteps=large_timesteps,
            random_seed=42
        )
        elapsed_time = time.time() - start_time

        # Check that it completed reasonably quickly
        self.assertLess(
            elapsed_time, 2.0,
            f"Matrix construction took {elapsed_time:.2f}s, should be < 2s")

        # Check results are valid
        self.assertEqual(risk_matrix.shape,
                         (len(large_base_risks), large_timesteps))
        self.assertTrue(np.all((risk_matrix >= 0) & (risk_matrix <= 1)))

    def test_matrix_reproducibility(self):
        """Test that matrix construction is reproducible with same seed."""
        matrix1 = build_temporal_risk_matrix(
            self.base_risks,
            n_timesteps=self.n_timesteps,
            random_seed=123
        )

        matrix2 = build_temporal_risk_matrix(
            self.base_risks,
            n_timesteps=self.n_timesteps,
            random_seed=123
        )

        np.testing.assert_array_equal(
            matrix1, matrix2,
            err_msg="Matrices should be identical with same random seed"
        )

    def test_matrix_seasonal_effects(self):
        """Test that seasonal effects are captured in the matrix."""
        # Build matrix with strong seasonal effects
        risk_matrix = build_temporal_risk_matrix(
            self.base_risks,
            n_timesteps=26,  # Half year
            seasonal_amplitude=0.3,
            seasonal_period=52,
            rho=0.95,  # High persistence to see seasonal pattern
            sigma=0.01,  # Low noise
            random_seed=42
        )

        # Calculate population mean over time
        population_means = np.mean(risk_matrix, axis=0)

        # Check that there's variation (indicating seasonal effects)
        variation = np.max(population_means) - np.min(population_means)
        self.assertGreater(variation, 0.01,
                           "Should see seasonal variation in population means")

        # Check that it's bounded
        self.assertLess(variation, 0.1,
                        "Seasonal variation should be reasonable")


if __name__ == '__main__':
    unittest.main()
