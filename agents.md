# Agents.md

## Overview
This markdown file defines the overall project structure and code style used in the project.

### Code repository
When asked to update the code repository structure, only include the folders, do not include individual files. Next to the folder name, give a short summary of its contents.

```text
football_analytics/
|-- .github/                  GitHub workflows and repository automation
|-- src/                       Core Python package and project source code
|   |-- api/                   API entrypoints and versioned endpoints
|   |-- 01-analysis/           Analysis workspace assets (phase 1)
|   |-- 02-analysis/           Analysis workspace assets (phase 2)
|   |-- 03-analysis/           Analysis workspace assets (phase 3)
|   |-- training/              Training and hyperparameter tuning pipelines
|   |-- evaluation/            Model evaluation workflows and reporting logic
|   |-- feature_engineering/   Feature transforms, OpenFE logic, and transformers
|   |-- model/                 Model wrappers and training-related data classes
|   |-- utils/                 Shared utilities and handlers used across modules
|   |-- data/                  Source-controlled data assets used by code under src
|-- tests/                     Unit and integration tests
|-- frontend/                  Frontend app/assets
|-- figures/                   Generated plots and figures
|-- mlruns/                    Local MLflow tracking runs
|-- mlartifacts/               Local MLflow artifact store
```


### Code style 
Follow the standards defined in the `pyproject.toml` file.
- Avoid writing unnecessary comments inline.
- Avoid defining functions within functions.
- Avoid implementing "fallbacks" unless explicitly prompted.
- Docstring format: Follow the autoDocstring format. 

### Tests
Unit tests are placed in `tests/` and the name convention is to use `test_` prefix.

### Common commands
- Run all tests: `pytest`
- Run one test module: `pytest tests/test_column_transformer.py`
- Lint and format (ruff): `ruff check .` and `ruff format .`

### Agent guardrails
- Prefer modifying modules in `src/feature_engineering/`, `src/model/`, and `src/utils/` instead of notebooks under analysis folders for production logic.
- Keep test updates close to behavior changes and use explicit test names describing what is validated.
- Avoid broad refactors unless requested; prioritize minimal, verifiable changes.
