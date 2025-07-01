# Healthcare AI Temporal Simulation Framework

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](https://github.com/Mdraugelis/pop-ml-simulator/actions)

A sophisticated temporal simulation framework for evaluating healthcare AI interventions with known ground truth. Enables comprehensive causal inference analysis of AI-guided healthcare interventions through realistic population modeling and intervention simulation.

## 🎯 The Challenge

Healthcare AI systems promise to revolutionize patient care by identifying high-risk individuals for early intervention. However, evaluating these systems presents a fundamental challenge: **we never observe the counterfactual**—what would have happened without the AI intervention.

### Real-World Scenarios

**Stroke Prevention Program**: An AI system predicts stroke risk within 12 months. When high-risk patients receive preventive interventions, we face an attribution problem: Did the intervention prevent the stroke, or was the patient never going to have one?

**Heart Failure Readmission Prevention**: AI flags high-risk patients for enhanced monitoring. We observe partial outcomes (some readmissions occur despite intervention) but face time-varying effects, spillover effects, and competing events.

**Solution**: This framework provides a controlled environment with **known ground truth** at every level, enabling validation of causal inference methods before real-world deployment.

## 🏗️ Framework Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 TEMPORAL SIMULATION FRAMEWORK               │
├─────────────────────────────────────────────────────────────┤
│ 1. Population Risk Modeling │ 2. Temporal Risk Dynamics     │
│ 3. Hazard-based Incidents   │ 4. ML Prediction Simulation   │
│ 5. Intervention Effects     │ 6. Causal Inference Analysis  │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

- **Risk Distribution**: Beta-distributed individual risks with controlled population prevalence
- **Temporal Dynamics**: AR(1) processes with seasonal patterns and external shocks
- **Hazard Modeling**: Survival analysis with competing risks and censoring
- **ML Simulation**: Controlled ML performance (PPV, sensitivity) with calibrated noise
- **Intervention Effects**: Known intervention effectiveness with realistic deployment
- **Causal Analysis**: RDD, DiD, ITS methods with ground truth validation

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Mdraugelis/pop-ml-simulator.git
cd pop-ml-simulator
pip install -r requirements.txt
```

### Basic Usage

```python
import numpy as np
from pop_ml_simulator import (
    assign_patient_risks,
    EnhancedTemporalRiskSimulator,
    IncidentGenerator,
    MLPredictionSimulator
)

# 1. Generate heterogeneous patient population
n_patients = 10000
base_risks = assign_patient_risks(
    n_patients, 
    annual_incident_rate=0.1,
    concentration=0.5,  # Controls risk heterogeneity
    random_seed=42
)

# 2. Add temporal dynamics
temporal_sim = EnhancedTemporalRiskSimulator(
    base_risks, 
    rho=0.9,  # Persistence
    sigma=0.1,  # Volatility
    seasonal_amplitude=0.2,
    seasonal_period=52  # Weekly cycles
)

# Add external shock (e.g., flu season)
temporal_sim.add_shock(
    time_step=26, 
    magnitude=1.5, 
    duration=8,
    affected_fraction=0.3
)

# 3. Generate realistic ML predictions
ml_sim = MLPredictionSimulator(
    target_sensitivity=0.8,
    target_ppv=0.3,
    random_seed=42
)

# 4. Simulate incidents and interventions
incident_gen = IncidentGenerator(timestep_duration=1/52)
true_labels = np.zeros(n_patients, dtype=int)

# Generate 6 months of training data
for week in range(26):
    incidents = incident_gen.generate_incidents(base_risks)
    true_labels |= incidents

# Generate calibrated ML predictions
predictions, binary_preds = ml_sim.generate_predictions(
    true_labels, base_risks
)

print(f"Population prevalence: {np.mean(true_labels):.1%}")
print(f"ML PPV achieved: {np.mean(true_labels[binary_preds == 1]):.1%}")
```

## 📊 Framework Capabilities

### 1. Population Risk Modeling

```python
# Beta-distributed risks with exact population control
base_risks = assign_patient_risks(
    n_patients=100000,
    annual_incident_rate=0.05,  # Exactly 5% population rate
    concentration=0.5,  # Right-skewed distribution
    random_seed=42
)

# Verify population constraint
assert abs(np.mean(base_risks) - 0.05) < 0.001
```

### 2. Temporal Risk Evolution

```python
# AR(1) process with seasonal patterns
temporal_sim = EnhancedTemporalRiskSimulator(
    base_risks,
    rho=0.9,  # High persistence
    sigma=0.1,  # Moderate volatility
    seasonal_amplitude=0.3,  # 30% seasonal variation
    seasonal_period=52  # Annual cycle
)

# Simulate time-varying risks
for week in range(52):
    temporal_sim.step()
    current_risks = temporal_sim.get_current_risks()
    # Use current_risks for incident generation
```

### 3. ML Performance Simulation

```python
# Generate ML model with exact performance targets
ml_sim = MLPredictionSimulator(
    target_sensitivity=0.85,
    target_ppv=0.40,
    calibration='sigmoid',
    random_seed=42
)

# Optimize noise parameters to hit targets
params = ml_sim.optimize_noise_parameters(true_labels, base_risks)
predictions, binary = ml_sim.generate_predictions(
    true_labels, base_risks,
    params['correlation'], params['scale']
)

# Validate performance
from pop_ml_simulator import evaluate_threshold_based
metrics = evaluate_threshold_based(true_labels, predictions, ml_sim.threshold)
print(f"Achieved sensitivity: {metrics['sensitivity']:.1%}")
print(f"Achieved PPV: {metrics['ppv']:.1%}")
```

### 4. Intervention Analysis

```python
# Evaluate different selection strategies
from pop_ml_simulator import evaluate_topk, optimize_alert_threshold

# Top-K selection (resource constrained)
topk_metrics = evaluate_topk(true_labels, predictions, k_percent=10)
print(f"Top 10% PPV: {topk_metrics['ppv']:.1%}")
print(f"Top 10% Sensitivity: {topk_metrics['sensitivity']:.1%}")

# Capacity-constrained optimization
alert_result = optimize_alert_threshold(
    predictions, true_labels,
    capacity_constraint=0.15,  # Can treat 15% of patients
    fatigue_weight=0.1
)
print(f"Optimal threshold: {alert_result['optimal_threshold']:.3f}")
print(f"Alert efficiency: {alert_result['efficiency']:.1%}")
```

## 📈 Advanced Features

### Risk Stratification Analysis

```python
from pop_ml_simulator import analyze_risk_stratified_performance

# Analyze performance across risk strata
strat_results = analyze_risk_stratified_performance(
    true_labels, predictions, base_risks, n_bins=5
)

print("Performance by Risk Quintile:")
print(strat_results[['risk_bin', 'prevalence', 'ppv', 'sensitivity']].round(3))
```

### Model Calibration Assessment

```python
from pop_ml_simulator import hosmer_lemeshow_test

# Test prediction calibration
hl_stat, p_value = hosmer_lemeshow_test(true_labels, predictions)
print(f"Hosmer-Lemeshow p-value: {p_value:.3f}")
print(f"Calibration {'PASSED' if p_value > 0.05 else 'FAILED'}")
```

### Competing Risks Modeling

```python
from pop_ml_simulator import CompetingRiskIncidentGenerator

# Model multiple competing outcomes
competing_gen = CompetingRiskIncidentGenerator(
    risk_types=['readmission', 'death', 'transfer'],
    timestep_duration=1/12  # Monthly
)

# Generate competing events
outcomes = competing_gen.generate_competing_incidents(
    readmission_risks=base_risks,
    death_risks=base_risks * 0.1,
    transfer_risks=base_risks * 0.05
)
```

## 🧮 Mathematical Foundations

### Risk Distribution
```
Individual risks ~ Beta(α, β) where:
α = concentration
β = α × (1/target_rate - 1)

Scaled to ensure: E[risks] = target_rate
```

### Temporal Evolution
```
risk_t = base_risk × temporal_modifier_t
temporal_modifier_t = ρ × modifier_{t-1} + (1-ρ) + ε_t
ε_t ~ N(0, σ²)
```

### Hazard Conversion
```
monthly_hazard = -ln(1 - annual_risk) / 12
timestep_prob = 1 - exp(-hazard × Δt)
```

### ML Simulation
```
predictions = calibration_fn(
    correlation × true_risk + (1-correlation) × noise + label_boost
)
```

## 📚 Interactive Notebooks

Comprehensive tutorials and examples:

- **[01_risk_distribution_exploration.ipynb](notebooks/01_risk_distribution_exploration.ipynb)**: Population risk modeling
- **[02_temporal_risk_dynamics.ipynb](notebooks/02_temporal_risk_dynamics.ipynb)**: Time-varying risk simulation  
- **[03_hazard_modeling.ipynb](notebooks/03_hazard_modeling.ipynb)**: Survival analysis and competing risks
- **[04_intervention_ml_simulation.ipynb](notebooks/04_intervention_ml_simulation.ipynb)**: ML prediction and intervention effects

## 🔬 Causal Inference Applications

This framework enables rigorous evaluation of causal inference methods:

### Regression Discontinuity Design (RDD)
- **Use case**: AI system treats patients above risk threshold
- **Validation**: Compare estimated vs. known intervention effect at threshold
- **Ground truth**: Exact effect size controlled by simulation parameters

### Difference-in-Differences (DiD)  
- **Use case**: Phased AI rollout across hospitals/clinics
- **Validation**: Parallel trends assumption with controlled temporal patterns
- **Ground truth**: Known treatment group assignment and intervention timing

### Interrupted Time Series (ITS)
- **Use case**: System-wide AI deployment with clear intervention point
- **Validation**: Level and slope changes against known intervention effects
- **Ground truth**: Precise intervention timing and controlled confounders

## 🎯 Real-World Applications

### AI Pilot Studies
Test different AI deployment strategies before real implementation:
```python
# Compare threshold-based vs. top-K selection
threshold_strategy = evaluate_threshold_based(labels, predictions, 0.3)
topk_strategy = evaluate_topk(labels, predictions, k_percent=15)

print(f"Threshold PPV: {threshold_strategy['ppv']:.1%}")
print(f"Top-K PPV: {topk_strategy['ppv']:.1%}")
```

### Resource Planning
Understand staffing needs for different intervention scenarios:
```python
# Analyze alert volume across different thresholds
thresholds = np.linspace(0.1, 0.9, 20)
alert_volumes = [
    evaluate_threshold_based(labels, predictions, t)['flag_rate']
    for t in thresholds
]
```

### Policy Evaluation
Simulate effects of changing treatment guidelines:
```python
# Test different intervention effectiveness levels
effectiveness_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
for effectiveness in effectiveness_levels:
    # Run intervention simulation with different effectiveness
    # Measure population-level impact
    pass
```

## 🏥 Healthcare Integration

### Clinical Decision Support
```python
# Generate risk displays for clinical use
from pop_ml_simulator import generate_risk_displays

patient_ids = [f"PT{i:04d}" for i in range(len(predictions))]
risk_displays = generate_risk_displays(patient_ids, predictions)

# Show top 10 highest risk patients
print(risk_displays.head(10)[['patient_id', 'risk_category', 'percentile']])
```

### Alert Optimization
```python
# Optimize alerts given capacity constraints
optimal_config = optimize_alert_threshold(
    predictions, true_labels,
    capacity_constraint=0.1,  # 10% capacity
    fatigue_weight=0.2  # Weight false positives heavily
)

print(f"Recommended threshold: {optimal_config['optimal_threshold']:.3f}")
print(f"Expected alert efficiency: {optimal_config['efficiency']:.1%}")
```

## 🧪 Testing and Validation

### Run Test Suite
```bash
# Run all tests
python tests/run_tests.py -k "not test_public_functions_are_decorated"

# Code quality checks  
flake8 src tests
mypy src

# Test notebooks
python tests/run_tests.py tests/test_notebooks.py -v
```

### Performance Benchmarks
The framework targets:
- **Speed**: 100K patients × 48 months < 60 seconds
- **Memory**: < 2GB for 100K population  
- **Accuracy**: ML performance within ±2% of targets
- **Calibration**: Hosmer-Lemeshow p-value > 0.05

## 📁 Project Structure

```
pop-ml-simulator/
├── src/pop_ml_simulator/
│   ├── risk_distribution.py     # Beta distribution risk modeling
│   ├── temporal_dynamics.py     # AR(1) temporal risk evolution
│   ├── hazard_modeling.py       # Survival analysis & competing risks
│   ├── ml_simulation.py         # ML prediction simulation
│   └── __init__.py
├── notebooks/                   # Interactive tutorials
├── tests/                       # Comprehensive test suite
├── requirements.txt             # Dependencies
├── CLAUDE.md                    # AI assistant guide
└── README.md                    # This file
```

## 🔧 Development

### Environment Setup
```bash
# Create Python 3.13.5 environment
source .python-version-setup.sh

# Install in development mode
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### Code Quality Standards
- **Type hints**: All functions use type annotations
- **Documentation**: Comprehensive docstrings with examples
- **Style**: flake8 compliant, line length ≤ 79 characters
- **Testing**: >90% test coverage
- **Performance**: Vectorized operations, minimal loops

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run quality checks: `flake8 src tests && mypy src`
5. Run test suite: `python tests/run_tests.py`
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📊 Performance Validation

### Framework Requirements Met
- ✅ **Accuracy**: ML models achieve target PPV/sensitivity within ±2%
- ✅ **Calibration**: Hosmer-Lemeshow tests pass (p > 0.05)
- ✅ **Speed**: Vectorized operations, optimized for large populations
- ✅ **Reproducibility**: All simulations use controlled random seeds
- ✅ **Flexibility**: Modular design supports diverse use cases

### Validation Results
```python
# Example validation output
Population size: 100,000
True prevalence: 10.2%
ML target: Sensitivity=80%, PPV=30%

Achieved performance:
Sensitivity: 79.7% (target: 80.0%) ✓
PPV: 29.9% (target: 30.0%) ✓
Calibration p-value: 0.67 ✓
Runtime: 14.2s ✓
```

## 📈 Future Roadmap

### Phase 1: Core Framework ✅
- [x] Population risk modeling with beta distributions
- [x] Temporal dynamics with AR(1) processes  
- [x] Hazard-based incident generation
- [x] ML prediction simulation with controlled performance
- [x] Comprehensive test suite and validation

### Phase 2: Causal Inference Methods 🚧
- [ ] Regression Discontinuity Design (RDD) implementation
- [ ] Difference-in-Differences (DiD) analysis tools
- [ ] Interrupted Time Series (ITS) methods
- [ ] Synthetic Control Method framework

### Phase 3: Advanced Features 📋
- [ ] Multi-hospital simulation capabilities
- [ ] Geographic and demographic stratification
- [ ] Cost-effectiveness analysis tools
- [ ] Real-time monitoring dashboards

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Healthcare AI research community
- Causal inference methodology experts  
- Open source scientific computing ecosystem (NumPy, SciPy, pandas)
- Clinical decision support researchers

## 📞 Support

- **Documentation**: [Notebooks](notebooks/) and inline documentation
- **Issues**: [GitHub Issues](https://github.com/Mdraugelis/pop-ml-simulator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Mdraugelis/pop-ml-simulator/discussions)

---

**Advancing healthcare AI through rigorous simulation and causal inference** 🏥🤖📊