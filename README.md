# pop-ml-sim: Temporal Healthcare AI Simulation Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A sophisticated simulation framework designed to optimize healthcare AI operations through evidence-based deployment strategies. By creating a "digital twin" of healthcare systems, we can evaluate AI implementation approaches and measure their operational impact before real-world deployment.

## 🎯 Core Innovation

**Dual-level architecture** (individual patients + hospital organizations) enables comprehensive operational assessment by modeling realistic healthcare AI deployment patterns and their systemic effects.

## 🏗️ Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SIMULATION FRAMEWORK                      │
├─────────────────────────────────────────────────────────────┤
│  Phase 1: Core Temporal Engine (Individual + Hospital)       │
│  Phase 2: AI Risk & Intervention Layer                       │
│  Phase 3: Causal Inference Analysis                          │
│  Phase 4: Multi-Method Validation                            │
└─────────────────────────────────────────────────────────────┘
```

### System Components

- **Population Generation**: Create individuals with demographics
- **Hospital Assignment**: Distribute across healthcare facilities  
- **Temporal Evolution**: Incremental updates with hazard-based events
- **AI Deployment Optimization**: AI-guided interventions and operational strategies
- **Outcome Measurement**: Track individual and hospital operational metrics
- **Impact Analysis**: Apply causal inference methods to measure operational effectiveness

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-org/pop-ml-sim.git
cd pop-ml-sim
pip install -r requirements.txt
```

### Basic Usage

```python
from pop_ml_sim import SimulationFramework
from pop_ml_sim.config import load_config

# Load configuration
config = load_config("configs/basic_simulation.yaml")

# Initialize framework
sim = SimulationFramework(config)

# Run simulation
results = sim.run()

# Analyze operational impact
sim.analyze_operational_effects(results)
```

## 📋 Core Data Structures

### PersonState
```python
@dataclass
class PersonState:
    person_id: int          # Unique identifier
    hospital_id: int        # Assigned hospital
    age: float             # Current age in years
    alive: bool = True     # Survival status
    had_incident: bool = False  # Incident history
```

### HospitalState
```python
@dataclass
class HospitalState:
    hospital_id: int
    size_category: str  # 'small', 'medium', 'large'
    baseline_quality: float
    ai_adoption_propensity: float
```

## 🧮 Mathematical Foundations

### Hazard Functions

**Incident hazard (monthly rate):**
```
λ_incident(t | age, X) = base_hazard × exp(age_effect × (age - 50) / 10)
```

**Mortality hazard:**
```
λ_mortality(t | age, incident) = base_hazard × exp(age_effect × (age - 50) / 10) × incident_multiplier
```

**Discrete time probability conversion:**
```
P(event) = 1 - exp(-λ × Δt)
```

## 🔬 Development Phases

### Phase 1: Core Temporal Engine ✅
- [x] Basic population and temporal mechanics
- [x] Configuration system with Hydra
- [x] Hospital-level aggregation
- [x] Performance optimization

### Phase 2: AI Operations & Intervention Layer 🔄
- [ ] ML model simulation with controlled operational performance
- [ ] AR(1) temporal consistency for risk scores
- [ ] Intervention assignment and resource allocation strategies
- [ ] Hospital-level AI adoption and operational policies

### Phase 3: Operational Impact Analysis 📋
- [ ] Regression Discontinuity Design (RDD) for threshold-based interventions
- [ ] Difference-in-Differences (DiD) for phased rollouts
- [ ] Interrupted Time Series (ITS) for policy changes
- [ ] Synthetic Control Method for comparative effectiveness

### Phase 4: Operations Research Tools 🎯
- [ ] ROI and cost-effectiveness calculators
- [ ] Operational assumption validators
- [ ] Implementation strategy comparison framework
- [ ] Real-time operational dashboards

## ⚙️ Configuration

### Population Configuration
```yaml
population:
  n_persons: 100000
  n_hospitals: 5
  age_distribution: uniform(18, 85)
  hospital_distribution:
    small: 0.5
    medium: 0.35
    large: 0.15
```

### Simulation Configuration
```yaml
simulation:
  pre_intervention_months: 24
  post_intervention_months: 12
  delta_t: 1.0  # months
  checkpoint_frequency: 6  # months
```

### Hazard Configuration
```yaml
hazards:
  incident:
    base_hazard: 0.001
    age_effect: 0.02
  mortality:
    base_hazard: 0.0001
    incident_multiplier: 2.0
```

## 🎯 Use Cases

1. **AI Deployment Planning**: Optimize implementation strategies and resource allocation
2. **Operational Readiness**: Assess staffing needs and workflow changes for AI interventions
3. **Policy Impact Assessment**: Simulate operational effects of guideline changes
4. **Performance Optimization**: Find optimal risk thresholds and intervention protocols
5. **Implementation Strategy Validation**: Compare operational approaches and their effectiveness
6. **Operations Training Platform**: Train teams on AI deployment and impact measurement

## 📊 Performance Targets

- Simulate 200k patients × 48 months < 120 seconds
- Memory usage < 2GB for 100k population
- Linear scaling with population size
- Support all major operational impact assessment methods

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v --cov=pop_ml_sim
```

Performance benchmarks:
```bash
python benchmarks/performance_test.py
```

## 📁 Project Structure

```
pop-ml-sim/
├── pop_ml_sim/
│   ├── core/               # Core simulation engine
│   ├── models/             # AI model operational simulation
│   ├── operations/         # Operational impact analysis methods
│   ├── config/             # Configuration management
│   └── utils/              # Utility functions
├── configs/                # YAML configuration files
├── tests/                  # Test suite
├── benchmarks/             # Performance benchmarks
├── examples/               # Usage examples
└── docs/                   # Documentation
```

## 🔧 Development

### Setup Development Environment
```bash
pip install -e ".[dev]"
pre-commit install
```

### Code Quality Standards
- Type hints on all functions
- Comprehensive docstrings
- Black formatting
- Flake8 compliance
- Test coverage > 40%

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 Output Formats

### Data Storage Options
- **Parquet**: Default for large datasets
- **CSV**: Human-readable exports
- **HDF5**: Complex hierarchical data
- **JSON**: Configuration and metadata

### Compression
- **Snappy** (default): Fast compression
- **Gzip**: Better compression ratio
- **LZ4**: Fastest compression

## 🤝 Operational Assessment Requirements

### Minimum Sample Sizes for Robust Analysis
- **DiD**: 20-30 hospitals (10-15 treatment, 20-30 control) for phased rollouts
- **Synthetic Control**: 3-5x donor units vs treated for comparative effectiveness
- **RDD**: Sufficient density around threshold for intervention protocols
- **ITS**: 24+ pre-intervention months for policy impact assessment

### Hospital Configuration
- Total: 30-50 hospitals minimum
- Size variation for heterogeneity
- Geographic/quality variation (future)

## 📚 Documentation

- [User Guide](docs/user_guide.md)
- [API Reference](docs/api_reference.md)
- [Configuration Guide](docs/configuration.md)
- [Operational Assessment Guide](docs/operational_guide.md)
- [Implementation Strategy Methods](docs/implementation_methods.md)
- [Performance Tuning](docs/performance.md)

## 📈 Success Metrics

- ✅ Simulate 200k patients × 48 months < 120 seconds
- ✅ Support all major operational impact assessment methods
- ✅ Achieve specified operational performance targets (±5%)
- ✅ Enable reproducible operational experiments
- ✅ Provide practical deployment value

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Healthcare AI operations community
- Implementation science methodologists
- Open source scientific computing ecosystem

## 📞 Support

- Documentation: [Read the Docs](https://pop-ml-sim.readthedocs.io/)
- Issues: [GitHub Issues](https://github.com/your-org/pop-ml-sim/issues)
- Discussions: [GitHub Discussions](https://github.com/your-org/pop-ml-sim/discussions)

---

**Built with ❤️ for advancing healthcare AI operations**
