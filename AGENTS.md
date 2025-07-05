# Guidelines for AI Agents

This repository contains Python code and Jupyter notebooks for simulating and evaluating healthcare interventions. Use this guide when contributing automated changes.
Always reference the design document to guide architecture, function and requirements `docs/Healthcare_Sim_Design.md/`.
## Project Structure
- `src/` – Core Python modules for the project, simulations and analysis.
- `notebooks/` – Jupyter notebooks used for data exploration and method experimentation. They may also contain embedded tests via `ipytest`.
- 'experiments' - Hold outputs from experimental runs
- 'configs' - Hold configuration files
- `tests/` – Unit tests covering the modules in `src/`. Includes a `run_tests.py` helper for running tests with coverage.
- Other files such as `config.yaml` and `requirements.txt` contain configuration and dependencies.

## Coding Conventions
- **Language**: Python 3.13.5
- **Libraries**: numpy, pandas, matplotlib, statsmodels and related scientific libraries. All dependencies are listed in `requirements.txt`.
- **Style**: Follow standard PEP8 conventions. Use snake_case for functions and variables and PascalCase for classes. Provide descriptive docstrings and type hints where practical.

## Testing Protocols
- Unit tests live in the `tests/` directory and are executed with `pytest`.
- Run the full suite with coverage using:
  ```bash
  python tests/run_tests.py
  ```
  or simply:
  ```bash
  pytest -v
  ```
- Notebooks may embed tests using `ipytest`. To run them inside a notebook:
  ```python
  import ipytest
  ipytest.run('-q')
  ```
- Write meaningful tests when modifying or adding functionality. Aim for good coverage and clear assertions.

## Logging (Optional)
- Public functions may optionally be decorated with @log_call from src/utils/logging.py for debugging and monitoring.
- INFO level records entry, exit, and runtime only (no payloads).
- DEBUG level additionally records sanitized input arguments and return values.
- Do not log raw PHI/PII; use redact() helper or customize as needed.
- Set log level via APP_LOG_LEVEL environment variable; default is CRITICAL (off).

## Pull Request Guidelines
- Provide a clear summary of changes and reference any related issues.
- Ensure all tests pass and coverage reports generate without errors.
- Keep PRs focused: avoid mixing unrelated changes.
- Use the repository’s GitHub Actions (`Python Tests` and `Run Tests`) as confirmation that checks pass.

## Programmatic Checks
Run these commands before opening a PR:
```bash
flake8 src tests            # style check (if flake8 is installed)
mypy src                    # optional static type checking
python tests/run_tests.py   # run unit tests with coverage
# Optional: pytest -q tests/test_log_wrappers.py   # checks if public functions have @log_call decorators
```

When notebooks contain tests via `ipytest`, execute the notebook cells to ensure those tests succeed as well.
