"""
Tests for hazard modeling module.
"""

import unittest
import numpy as np
from pop_ml_simulator.hazard_modeling import (
    annual_risk_to_hazard,
    hazard_to_timestep_probability,
    IncidentGenerator,
    CompetingRiskIncidentGenerator
)


class TestHazardFunctions(unittest.TestCase):
    """Test cases for hazard conversion functions."""

    def test_annual_risk_to_hazard_scalar(self):
        """Test risk to hazard conversion with scalar input."""
        # Test known values
        self.assertAlmostEqual(annual_risk_to_hazard(0.0), 0.0)
        self.assertAlmostEqual(
            annual_risk_to_hazard(0.1), -np.log(0.9), places=6)

        # For small risks, hazard ≈ risk
        small_risk = 0.01
        hazard = annual_risk_to_hazard(small_risk)
        self.assertAlmostEqual(hazard, small_risk, places=3)

    def test_annual_risk_to_hazard_array(self):
        """Test risk to hazard conversion with array input."""
        risks = np.array([0.0, 0.1, 0.5, 0.9])
        hazards = annual_risk_to_hazard(risks)

        self.assertEqual(hazards.shape, risks.shape)
        self.assertAlmostEqual(hazards[0], 0.0)
        self.assertAlmostEqual(hazards[1], -np.log(0.9))
        self.assertAlmostEqual(hazards[2], -np.log(0.5))
        self.assertAlmostEqual(hazards[3], -np.log(0.1))

    def test_annual_risk_to_hazard_edge_cases(self):
        """Test risk to hazard conversion with edge cases."""
        # Test that risks >= 1.0 are handled gracefully
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Test risk = 1.0 (should warn and clip)
            hazard = annual_risk_to_hazard(1.0)
            self.assertTrue(len(w) >= 1)
            self.assertTrue(np.isfinite(hazard))

            # Test risk > 1.0 (should warn and clip)
            hazard = annual_risk_to_hazard(1.5)
            self.assertTrue(np.isfinite(hazard))

            # Test array with some values >= 1.0
            risks = np.array([0.5, 1.0, 1.2])
            hazards = annual_risk_to_hazard(risks)
            self.assertTrue(np.all(np.isfinite(hazards)))
            self.assertEqual(len(hazards), 3)

    def test_hazard_to_timestep_probability_scalar(self):
        """Test hazard to probability conversion with scalar input."""
        hazard = 0.1

        # Daily
        daily_prob = hazard_to_timestep_probability(hazard, 1/365)
        # Should be close to linear approximation
        self.assertLess(daily_prob, hazard / 365 * 1.01)

        # Annual
        annual_prob = hazard_to_timestep_probability(hazard, 1.0)
        expected = 1 - np.exp(-hazard)
        self.assertAlmostEqual(annual_prob, expected, places=6)

    def test_hazard_to_timestep_probability_array(self):
        """Test hazard to probability conversion with array input."""
        hazards = np.array([0.1, 0.5, 1.0])
        timestep = 1/52  # Weekly

        probs = hazard_to_timestep_probability(hazards, timestep)

        self.assertEqual(probs.shape, hazards.shape)
        self.assertTrue(np.all(probs >= 0))
        self.assertTrue(np.all(probs <= 1))

        # Should be monotonic
        self.assertTrue(np.all(np.diff(probs) > 0))

    def test_round_trip_conversion(self):
        """Test that conversions are consistent."""
        annual_risk = 0.1

        # Convert to hazard and back for different timesteps
        hazard = annual_risk_to_hazard(annual_risk)

        # Annual timestep should recover original risk
        recovered_risk = hazard_to_timestep_probability(hazard, 1.0)
        self.assertAlmostEqual(recovered_risk, annual_risk, places=6)

        # Multiple timesteps should compound correctly
        weekly_prob = hazard_to_timestep_probability(hazard, 1/52)
        annual_from_weekly = 1 - (1 - weekly_prob)**52
        self.assertAlmostEqual(annual_from_weekly, annual_risk, places=4)


class TestIncidentGenerator(unittest.TestCase):
    """Test cases for IncidentGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_patients = 1000
        self.annual_risks = np.full(self.n_patients, 0.1)  # 10% annual risk

    def test_initialization(self):
        """Test generator initialization."""
        gen = IncidentGenerator(timestep_duration=1/52)

        self.assertEqual(gen.timestep_duration, 1/52)
        self.assertEqual(len(gen.incident_history), 0)
        self.assertIsNone(gen.cumulative_incidents)

    def test_generate_incidents_basic(self):
        """Test basic incident generation."""
        gen = IncidentGenerator(timestep_duration=1/52)

        incidents = gen.generate_incidents(self.annual_risks)

        # Check output
        self.assertEqual(len(incidents), self.n_patients)
        self.assertEqual(incidents.dtype, bool)

        # Should have some incidents but not too many for one week
        n_incidents = np.sum(incidents)
        self.assertGreater(n_incidents, 0)
        self.assertLess(n_incidents, 50)  # Roughly 10%/52 ≈ 0.2% per week

        # Check history updated
        self.assertEqual(len(gen.incident_history), 1)
        self.assertIsNotNone(gen.cumulative_incidents)

    def test_generate_incidents_with_intervention(self):
        """Test incident generation with intervention."""
        gen = IncidentGenerator(timestep_duration=1/52)

        # Half get intervention
        intervention_mask = np.zeros(self.n_patients, dtype=bool)
        intervention_mask[:500] = True
        intervention_effectiveness = 0.5  # 50% risk reduction

        # Run for multiple weeks
        for _ in range(52):
            gen.generate_incidents(
                self.annual_risks,
                intervention_mask,
                intervention_effectiveness
            )

        # Check intervention effect
        control_rate = np.mean(
            gen.cumulative_incidents[~intervention_mask] > 0)
        treated_rate = np.mean(gen.cumulative_incidents[intervention_mask] > 0)

        # Treated should have lower rate
        self.assertLess(treated_rate, control_rate)

        # Should be approximately 50% reduction
        relative_reduction = (control_rate - treated_rate) / control_rate
        self.assertAlmostEqual(relative_reduction, 0.5, delta=0.1)

    def test_cumulative_incidence(self):
        """Test cumulative incidence calculation."""
        gen = IncidentGenerator(timestep_duration=1/52)

        # Initially zero
        self.assertEqual(gen.get_cumulative_incidence(), 0.0)

        # Simulate full year
        prev_incidence = 0.0  # Initialize for first iteration
        for week in range(52):
            gen.generate_incidents(self.annual_risks)

            # Should be monotonic increasing
            current_incidence = gen.get_cumulative_incidence()
            if week > 0:
                self.assertGreaterEqual(current_incidence, prev_incidence)
            prev_incidence = current_incidence

        # Final incidence should be close to 10%
        final_incidence = gen.get_cumulative_incidence()
        self.assertAlmostEqual(final_incidence, 0.1, delta=0.02)

    def test_reset(self):
        """Test reset functionality."""
        gen = IncidentGenerator()

        # Generate some incidents
        gen.generate_incidents(self.annual_risks)
        self.assertGreater(len(gen.incident_history), 0)
        self.assertIsNotNone(gen.cumulative_incidents)

        # Reset
        gen.reset()
        self.assertEqual(len(gen.incident_history), 0)
        self.assertIsNone(gen.cumulative_incidents)

    def test_zero_risk(self):
        """Test with zero risk patients."""
        gen = IncidentGenerator()
        zero_risks = np.zeros(100)

        incidents = gen.generate_incidents(zero_risks)

        # Should have no incidents
        self.assertEqual(np.sum(incidents), 0)
        self.assertEqual(gen.get_cumulative_incidence(), 0.0)


class TestCompetingRiskIncidentGenerator(unittest.TestCase):
    """Test cases for CompetingRiskIncidentGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_patients = 1000
        self.risks_dict = {
            'readmission': np.full(self.n_patients, 0.3),  # 30% annual
            'death': np.full(self.n_patients, 0.1)  # 10% annual
        }

    def test_initialization(self):
        """Test competing risk generator initialization."""
        gen = CompetingRiskIncidentGenerator(
            timestep_duration=1/52,
            event_types=['readmission', 'death']
        )

        self.assertEqual(gen.event_types, ['readmission', 'death'])
        self.assertEqual(len(gen.event_history), 2)
        self.assertEqual(len(gen.cumulative_events), 2)
        self.assertIsNone(gen.censored)

    def test_generate_competing_incidents_basic(self):
        """Test basic competing incident generation."""
        gen = CompetingRiskIncidentGenerator(
            event_types=['readmission', 'death']
        )

        events = gen.generate_competing_incidents(self.risks_dict)

        # Check output structure
        self.assertIn('readmission', events)
        self.assertIn('death', events)
        self.assertEqual(len(events['readmission']), self.n_patients)
        self.assertEqual(len(events['death']), self.n_patients)

        # Check histories updated
        self.assertEqual(len(gen.event_history['readmission']), 1)
        self.assertEqual(len(gen.event_history['death']), 1)

    def test_competing_risks_exclusivity(self):
        """Test that once an event occurs, patient is no longer at risk."""
        gen = CompetingRiskIncidentGenerator(
            event_types=['readmission', 'death']
        )

        # Use very high risks to ensure events
        high_risks = {
            'readmission': np.full(10, 0.99),
            'death': np.full(10, 0.99)
        }

        # Run for several timesteps
        for _ in range(10):
            gen.generate_competing_incidents(high_risks)

        # Each patient should have at most one event type
        readmit_occurred = gen.cumulative_events['readmission'] > 0
        death_occurred = gen.cumulative_events['death'] > 0

        # No patient should have both events
        both_events = readmit_occurred & death_occurred
        self.assertEqual(np.sum(both_events), 0)

    def test_censoring(self):
        """Test censoring functionality."""
        gen = CompetingRiskIncidentGenerator(
            event_types=['readmission', 'death']
        )

        # Run with censoring
        for _ in range(52):
            gen.generate_competing_incidents(
                self.risks_dict,
                censoring_prob=0.01  # 1% per week
            )

        # Should have some censored patients
        self.assertGreater(np.sum(gen.censored), 0)

        # Censored patients should not have events after censoring
        cumulative = gen.get_cumulative_incidence_competing()
        self.assertIn('censored', cumulative)
        self.assertGreater(cumulative['censored'], 0)

    def test_cumulative_incidence_competing(self):
        """Test cumulative incidence calculation for competing risks."""
        gen = CompetingRiskIncidentGenerator(
            event_types=['readmission', 'death']
        )

        # Simulate full year
        for _ in range(52):
            gen.generate_competing_incidents(self.risks_dict)

        cumulative = gen.get_cumulative_incidence_competing()

        # Check all event types included
        self.assertIn('readmission', cumulative)
        self.assertIn('death', cumulative)
        self.assertIn('censored', cumulative)

        # Readmission should be higher than death (30% vs 10% risk)
        self.assertGreater(cumulative['readmission'], cumulative['death'])

        # Total events should be less than sum (due to competition)
        total_events = cumulative['readmission'] + cumulative['death']
        self.assertLess(total_events, 0.4)  # Less than 30% + 10%

        # Should be reasonable values
        self.assertAlmostEqual(cumulative['readmission'], 0.25, delta=0.05)
        self.assertAlmostEqual(cumulative['death'], 0.08, delta=0.03)


if __name__ == '__main__':
    unittest.main()
