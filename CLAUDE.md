# Claude Assistant Guide

This guide helps AI assistants understand the project structure and development patterns for the Population ML Simulator.

## Project Overview
A healthcare simulation framework for modeling patient populations, risk distributions, and intervention effects using:
- Beta distributions for patient risk modeling
- AR(1) processes for temporal risk dynamics
- Hazard functions and survival analysis
- Competing risks and censoring models
- ML prediction simulation with controlled performance

Always reference the design document to guide architecture, function and requirements `docs/Healthcare_Sim_Design.md/`

## Key Commands
```bash
# Run fast core tests (excludes expensive ML optimization and notebooks)
python tests/run_tests.py -k "not test_public_functions_are_decorated and not test_notebook"

# Run all tests including expensive ones (CI/local comprehensive testing)
python tests/run_tests.py -k "not test_public_functions_are_decorated" 

# Run tests in parallel for faster execution (requires pytest-xdist)
python tests/run_tests.py -n auto

# Code quality checks
flake8 src tests
mypy src

# Test notebooks only
python tests/run_tests.py tests/test_notebooks.py -v

# Activate Python 3.13.5 environment
source .python-version-setup.sh
```

## Project Structure
```
src/pop_ml_simulator/
├── risk_distribution.py    # Beta distribution risk modeling
├── temporal_dynamics.py    # AR(1) temporal risk evolution
├── hazard_modeling.py      # Survival analysis & competing risks
├── ml_simulation.py        # ML prediction simulation
└── __init__.py

notebooks/
├── 01_risk_distribution_exploration.ipynb
├── 02_temporal_risk_dynamics.ipynb
├── 03_hazard_modeling.ipynb
└── 04_intervention_ml_simulation.ipynb

tests/
├── test_risk_distribution.py
├── test_temporal_dynamics.py
├── test_hazard_modeling.py
├── test_ml_simulation.py
├── test_notebooks.py
└── run_tests.py            # Test runner with coverage
```

## Development Patterns

### Module Imports
Notebooks require path setup for module imports:
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), 'src'))
```

### Common Issues & Solutions
1. **ModuleNotFoundError in notebooks**: Add path setup (see above)
2. **RuntimeWarning for log(0)**: Clip values and add warnings for edge cases
3. **GitHub Actions import errors**: Update `tests/run_tests.py` to add src to path
4. **Missing dependencies**: Check `requirements.txt` includes all notebook dependencies
5. **ML performance targets not met**: Increase n_iterations in optimize_noise_parameters()
6. **Poor calibration**: Check Hosmer-Lemeshow p-value, adjust noise parameters

### Code Style
- Follow PEP8 conventions
- Line length: 79 characters max
- Use type hints where practical
- Logging decorators are optional (not mandatory)

### Testing Requirements
- All public functions should have unit tests
- Notebooks must execute without errors
- **Performance-first testing**: Expensive tests marked with @pytest.mark.skip for CI speed
- Fix all flake8 and mypy issues before PRs
- **Fast core test suite**: ~20 most expensive tests skipped for development speed
- **Full test suite**: Available for comprehensive validation when needed
- Use shared fixtures from conftest.py to avoid redundant setup

### Test Performance Strategy
- **Fast tests** (development): Core functionality tested in <2 minutes
- **Skipped tests**: ML optimization, benchmarking, integration, temporal validation
- **Full tests** (release): All tests including expensive ones for comprehensive coverage

### Git Workflow
1. Create feature branch: `git checkout -b feature/description`
2. Run flake8, mypy, all unit tests and test notebooks
3. Create PR with comprehensive description
4. Ensure GitHub Actions pass

## Recent Architecture Decisions
- Refactored from pickle serialization to Python modules (Issue #14)
- Consolidated dependencies into single `requirements.txt`
- Made logging optional rather than mandatory
- Added comprehensive notebook testing to CI/CD
- Implemented ML simulation framework (Issue #19) with controlled performance
- Optimized test performance (Issue #46): reduced dataset sizes, shared fixtures, parallel execution

## Environment
- Python: 3.13.5
- Key dependencies: numpy, pandas, scipy, matplotlib, seaborn
- Configuration: Hydra/OmegaConf for experiment management
