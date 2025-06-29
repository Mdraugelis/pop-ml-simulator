"""
Tests for risk distribution module.
"""

import unittest
import numpy as np
from pop_ml_simulator.risk_distribution import assign_patient_risks, simulate_annual_incidents


class TestRiskDistribution(unittest.TestCase):
    """Test cases for risk distribution functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_patients = 1000
        self.annual_incident_rate = 0.1
        self.concentration = 0.5
        self.random_seed = 42
    
    def test_assign_patient_risks_basic(self):
        """Test basic functionality of assign_patient_risks."""
        risks = assign_patient_risks(
            self.n_patients,
            self.annual_incident_rate,
            self.concentration,
            self.random_seed
        )
        
        # Check output shape
        self.assertEqual(len(risks), self.n_patients)
        
        # Check all risks are in valid range
        self.assertTrue(np.all(risks >= 0))
        self.assertTrue(np.all(risks <= 0.99))
        
        # Check mean is close to target
        self.assertAlmostEqual(np.mean(risks), self.annual_incident_rate, places=3)
    
    def test_assign_patient_risks_different_rates(self):
        """Test with different annual incident rates."""
        for rate in [0.01, 0.05, 0.1, 0.2, 0.5]:
            risks = assign_patient_risks(1000, rate, random_seed=42)
            # Higher rates may have more deviation due to clipping at 0.99
            places = 3 if rate < 0.3 else 2
            self.assertAlmostEqual(np.mean(risks), rate, places=places,
                                   msg=f"Failed for rate {rate}")
    
    def test_assign_patient_risks_concentration(self):
        """Test that concentration parameter affects heterogeneity."""
        # Lower concentration should give more heterogeneous risks
        risks_low = assign_patient_risks(10000, 0.1, concentration=0.1, random_seed=42)
        risks_high = assign_patient_risks(10000, 0.1, concentration=2.0, random_seed=42)
        
        # Check that low concentration has higher variance
        self.assertGreater(np.std(risks_low), np.std(risks_high))
        
        # Both should still have correct mean
        self.assertAlmostEqual(np.mean(risks_low), 0.1, places=2)
        self.assertAlmostEqual(np.mean(risks_high), 0.1, places=2)
    
    def test_assign_patient_risks_reproducibility(self):
        """Test that random seed ensures reproducibility."""
        risks1 = assign_patient_risks(100, 0.1, random_seed=42)
        risks2 = assign_patient_risks(100, 0.1, random_seed=42)
        
        np.testing.assert_array_equal(risks1, risks2)
    
    def test_assign_patient_risks_edge_cases(self):
        """Test edge cases."""
        # Very low rate
        risks = assign_patient_risks(1000, 0.001, random_seed=42)
        self.assertAlmostEqual(np.mean(risks), 0.001, places=4)
        
        # Very high rate
        risks = assign_patient_risks(1000, 0.9, random_seed=42)
        # High rates may deviate more due to clipping at 0.99
        self.assertAlmostEqual(np.mean(risks), 0.9, delta=0.02)
        
        # Single patient
        risk = assign_patient_risks(1, 0.1, random_seed=42)
        self.assertEqual(len(risk), 1)
    
    def test_simulate_annual_incidents_basic(self):
        """Test basic functionality of simulate_annual_incidents."""
        risks = assign_patient_risks(1000, 0.1, random_seed=42)
        counts = simulate_annual_incidents(risks, n_simulations=100, random_seed=42)
        
        # Check output shape
        self.assertEqual(len(counts), 100)
        
        # Check all counts are non-negative integers
        self.assertTrue(np.all(counts >= 0))
        self.assertTrue(np.all(counts <= len(risks)))
        
        # Check mean rate is close to expected
        mean_rate = np.mean(counts) / len(risks)
        self.assertAlmostEqual(mean_rate, 0.1, places=2)
    
    def test_simulate_annual_incidents_deterministic(self):
        """Test with deterministic risk values."""
        # All patients with 0 risk
        risks_zero = np.zeros(100)
        counts = simulate_annual_incidents(risks_zero, n_simulations=10)
        np.testing.assert_array_equal(counts, np.zeros(10))
        
        # All patients with 1.0 risk
        risks_one = np.ones(100) * 0.99  # Max allowed is 0.99
        counts = simulate_annual_incidents(risks_one, n_simulations=10)
        # Should be very close to all patients having incidents
        self.assertTrue(np.all(counts >= 95))
    
    def test_simulate_annual_incidents_variance(self):
        """Test that simulation variance matches theoretical expectations."""
        n_patients = 10000
        risks = assign_patient_risks(n_patients, 0.1, random_seed=42)
        counts = simulate_annual_incidents(risks, n_simulations=1000, random_seed=42)
        
        # For binomial, variance = n * p * (1-p) for each patient
        # Total variance is sum of individual variances
        expected_var = np.sum(risks * (1 - risks))
        observed_var = np.var(counts)
        
        # Should be within reasonable tolerance
        self.assertAlmostEqual(observed_var, expected_var, delta=expected_var * 0.1)
    
    def test_risk_distribution_properties(self):
        """Test statistical properties of risk distribution."""
        risks = assign_patient_risks(10000, 0.1, concentration=0.5, random_seed=42)
        
        # Should be right-skewed
        from scipy import stats
        skewness = stats.skew(risks)
        self.assertGreater(skewness, 0, "Distribution should be right-skewed")
        
        # Median should be less than mean for right-skewed distribution
        self.assertLess(np.median(risks), np.mean(risks))
        
        # Check that a small fraction of patients have high risk
        high_risk_fraction = np.mean(risks > 0.3)
        self.assertLess(high_risk_fraction, 0.2, 
                        "Less than 20% should be high risk (>30%)")


if __name__ == '__main__':
    unittest.main()