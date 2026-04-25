# Agents.md

## Overview
This markdown file defines the overall project structure and code style used in the project.

### Code repository
```text
football_analytics/
|-- agents.md
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- conftest.py
|-- api/
|   |-- main.py
|   `-- v1/
|-- feature_engineering/
|   |-- ColumnTransformer.py
|   |-- OpenFETransformations.py
|   |-- RowWiseTransformations.py
|   |-- OpenFE/
|   `-- __init__.py
|-- model/
|   |-- train.py
|   |-- eval.py
|   |-- nested_cv_eval.py
|   |-- tuning.py
|   `-- data_classes.py
|-- utils/
|   |-- preprocessing_handler.py
|   |-- feature_engineering_handler.py
|   |-- events_handler.py
|   |-- passes_handler.py
|   |-- shots_handler.py
|   |-- gmm_handler.py
|   |-- pitch_utils.py
|   |-- statics.py
|   `-- utils.py
|-- tests/
|   `-- test_column_transformer.py
|-- frontend/
|-- data/
|-- figures/
|-- 01-analysis/
`-- 02-analysis/
```


### Code style 
Follow the standards defined in the `pyproject.toml` file.


### Tests
Unit tests are placed in `tests/` and the name convention is to use `test_` prefix.

### Common commands
- Run all tests: `pytest`
- Run one test module: `pytest tests/test_column_transformer.py`
- Lint and format (ruff): `ruff check .` and `ruff format .`

### Agent guardrails
- Prefer modifying modules in `feature_engineering/`, `model/`, and `utils/` instead of notebooks under analysis folders for production logic.
- Keep test updates close to behavior changes and use explicit test names describing what is validated.
- Avoid broad refactors unless requested; prioritize minimal, verifiable changes.
