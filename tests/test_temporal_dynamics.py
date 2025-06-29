"""
Tests for temporal dynamics module.
"""

import unittest
import numpy as np
from pop_ml_simulator.temporal_dynamics import (
    simulate_ar1_process,
    TemporalRiskSimulator,
    EnhancedTemporalRiskSimulator
)


class TestAR1Process(unittest.TestCase):
    """Test cases for AR(1) process simulation."""

    def test_simulate_ar1_basic(self):
        """Test basic AR(1) simulation."""
        n_timesteps = 100
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
        n_timesteps = 1000
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

        n_steps = 52
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
        sim.simulate(100)

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


if __name__ == '__main__':
    unittest.main()
