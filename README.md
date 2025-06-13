# pop-ml-simulator

**Population-based Machine Learning Performance Simulator for Healthcare**

A Python framework that combines population-level disease simulation with controlled ML model performance, enabling researchers to generate synthetic patient cohorts with precisely calibrated prediction models.

## 🎯 What Makes This Different

Unlike traditional synthetic data generators that create realistic data and see what ML performance emerges, `pop-ml-simulator` **reverse-engineers** the problem: you specify your target ML performance metrics (PPV, Sensitivity, etc.) and the simulator generates a population and model that achieves those targets.

## 🚀 Key Features

- **Population Simulation**: Age-stratified cohorts with realistic disease incidence using hazard-based modeling
- **ML Performance Control**: Generate models that hit exact PPV and Sensitivity targets
- **Flexible Risk Models**: Switch between Beta distribution and hazard-based approaches
- **Vectorized Operations**: Optimized for large populations using NumPy and Numba
- **Healthcare Focus**: Built specifically for clinical prediction scenarios (stroke, cancer screening, etc.)
- **Python Native**: Seamless integration with scikit-learn, pandas, and ML workflows

## 🔬 Use Cases

- **Algorithm Validation**: Test ML pipelines against known ground truth
- **Performance Benchmarking**: Compare models under controlled conditions  
- **Threshold Optimization**: Simulate different decision thresholds and their outcomes
- **Clinical Trial Design**: Model patient populations before real studies
- **Healthcare AI Research**: Generate datasets for method development
- **Regulatory Validation**: Demonstrate ML performance under various scenarios

## 📊 Example: Stroke Risk Prediction

```python
from pop_ml_simulator import PopulationSimulator, BetaRiskModel

# Create population with age-dependent stroke risk
sim = PopulationSimulator(
    population_size=10000,
    annual_incidence_rate=0.02,
    num_years=5,
    age_dependent_risk=True
)

# Generate ML model with target performance
model = BetaRiskModel(target_ppv=0.75, target_sensitivity=0.80)

# Run simulation
sim.run_simulation()
results = sim.run_risk_prediction_with_analysis(model)

print(f"Achieved PPV: {results['mean_ppv']:.3f}")
print(f"Achieved Sensitivity: {results['mean_sensitivity']:.3f}")
```

## 🛠️ Installation

```bash
pip install pop-ml-simulator
```

Or for development:
```bash
git clone https://github.com/[username]/pop-ml-simulator.git
cd pop-ml-simulator
pip install -e .
```

## 📈 How It Works

1. **Population Generation**: Creates synthetic patients with realistic age distributions and risk factors
2. **Hazard Modeling**: Simulates disease incidence using time-dependent hazard functions
3. **Risk Score Calibration**: Generates ML predictions that achieve specified performance targets
4. **Performance Validation**: Tracks PPV, Sensitivity, and other metrics over time

## 🎛️ Flexible Architecture

- **Hazard Functions**: Pluggable disease models (stroke, cancer, heart disease)
- **Risk Models**: Beta distribution or hazard-based approaches
- **Performance Targets**: PPV, Sensitivity, F1-score, or custom metrics
- **Population Parameters**: Age distributions, mortality rates, risk factors

## 🔬 Validation & Research

Built with healthcare AI research in mind:
- Ground truth validation against real clinical data patterns
- Statistical methods based on survival analysis and clinical epidemiology  
- Performance metrics aligned with clinical practice (PPV, NPV, Sensitivity, Specificity)
- Integration with standard ML evaluation frameworks

## 📚 Documentation

- [Getting Started Guide](docs/getting_started.md)
- [Examples & Tutorials](examples/)
- [Clinical Use Cases](docs/clinical_examples.md)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Inspired by [Synthea](https://github.com/synthetichealth/synthea), [simtrial](https://github.com/Merck/simtrial), and the broader synthetic healthcare data community.

## 🏷️ Keywords

`synthetic-data` `healthcare-ml` `population-simulation` `model-calibration` `clinical-prediction` `precision-recall` `healthcare-ai` `risk-prediction` `epidemiology` `biostatistics`
