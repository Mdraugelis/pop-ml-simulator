# Claude Assistant Guide

This guide helps AI assistants understand the project structure and development patterns for the Population ML Simulator.

## Project Overview
A healthcare simulation framework for modeling patient populations, risk distributions, and intervention effects using:
- Beta distributions for patient risk modeling
- AR(1) processes for temporal risk dynamics
- Hazard functions and survival analysis
- Competing risks and censoring models

## Key Commands
```bash
# Run tests (excluding optional logging decorator test)
python tests/run_tests.py -k "not test_public_functions_are_decorated"

# Code quality checks
flake8 src tests
mypy src

# Test notebooks
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
└── __init__.py

notebooks/
├── 01_risk_distribution_exploration.ipynb
├── 02_temporal_risk_dynamics.ipynb
└── 03_hazard_modeling.ipynb

tests/
├── test_risk_distribution.py
├── test_temporal_dynamics.py
├── test_hazard_modeling.py
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

### Code Style
- Follow PEP8 conventions
- Line length: 79 characters max
- Use type hints where practical
- Logging decorators are optional (not mandatory)

### Testing Requirements
- All public functions should have unit tests
- Notebooks must execute without errors
- Maintain >90% test coverage
- Fix all flake8 and mypy issues before PRs

### Git Workflow
1. Create feature branch: `git checkout -b feature/description`
2. Run all tests and quality checks
3. Create PR with comprehensive description
4. Ensure GitHub Actions pass

## Recent Architecture Decisions
- Refactored from pickle serialization to Python modules (Issue #14)
- Consolidated dependencies into single `requirements.txt`
- Made logging optional rather than mandatory
- Added comprehensive notebook testing to CI/CD

## Environment
- Python: 3.13.5
- Key dependencies: numpy, pandas, scipy, matplotlib, seaborn
- Configuration: Hydra/OmegaConf for experiment management