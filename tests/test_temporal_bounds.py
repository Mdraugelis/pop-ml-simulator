import numpy as np
import pytest
from pop_ml_simulator import EnhancedTemporalRiskSimulator


class TestTemporalRiskBounds:

    def setup_method(self):
        np.random.seed(42)
        self.base_risks = np.random.beta(2, 18, 500)
        self.simulator = EnhancedTemporalRiskSimulator(
            self.base_risks,
            temporal_bounds=(0.2, 2.5),
            max_risk_threshold=0.95,
        )

    def test_no_risk_exceeds_one(self):
        for _ in range(50):
            risks = self.simulator.step()
            assert np.max(risks) <= 1.0

    def test_population_rate_preservation(self):
        target = np.mean(self.base_risks)
        for _ in range(20):
            risks = self.simulator.step()
            assert abs(np.mean(risks) - target) <= 0.02

    def test_temporal_bounds_enforced(self):
        for _ in range(10):
            self.simulator.step()
            assert np.all(self.simulator.temporal_modifiers >= 0.2)
            assert np.all(self.simulator.temporal_modifiers <= 2.5)

    def test_shock_validation(self):
        with pytest.raises(ValueError):
            self.simulator.add_shock(
                5, magnitude=5.0, duration=3, affected_fraction=0.5
            )
        with pytest.raises(ValueError):
            self.simulator.add_shock(
                5, magnitude=0.05, duration=3, affected_fraction=0.5
            )
