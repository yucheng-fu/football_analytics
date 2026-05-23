# Football analytics ⚽
A Python-based football analytics project for analysing the 2018 FIFA World cup using publicly available event data provided by [Statsbomb](https://github.com/statsbomb/open-data).

The project currently contains the following analyses:
- 01-analysis: Events, xG and possession patterns: An introduction to event-based data in football 
- 02-analysis: Quantifying quantifying passing difficulty with gradient-boosted tree models: machine learning applied to event data

## 📊 Analysis Overview

| Analysis | Description | Notebook | Blog |
|----------|-------------|----------|------|
| **01 — Event & xG Analysis** | Events, xG and possession pattern analysis using StatsBomb data | [📓 Notebook](https://github.com/yucheng-fu/football_analytics/blob/main/src/01-analysis/01-analysis.ipynb) | [📝 Blog](https://yucheng-fu.github.io/blog/2025/wc2018-part1/) |
| **02 — Pass Difficulty Modeling** | Gradient-boosted models for quantifying pass difficulty | [📓 Notebook](https://github.com/yucheng-fu/football_analytics/blob/main/src/02-analysis/02-analysis.ipynb) | TBD |


You can also check out the interactive pass classifier [here](https://yucheng-fu.github.io/football-analytics/).

## 💃 Installation 
Clone the repository:
```bash
git clone https://github.com/yucheng-fu/football_analytics.git
cd football_analytics
```

Install `uv` package manager. Please refer to their [documentation](https://docs.astral.sh/uv/getting-started/installation/)

Create virtual environment and install packages
```bash
uv sync
```

## 🔬 Tests, linting and formatting 
Run all tests
```bash
pytest
```

Run specific test
```bash
pytest tests/test_column_transformer.py
```

Run linting and formatting with `ruff`
```bash
ruff check . 
ruff check . --select I --fix
ruff format .
```
## 🚀 Deployment
For deployment of FastAPI backend, see [README](https://github.com/yucheng-fu/football_analytics/blob/main/src/api/README.md)

For deployment of frontend, see [README](https://github.com/yucheng-fu/football_analytics/blob/main/frontend/README.md)

## 🛠️ Tech stack 
**Backend:** Python, FastAPI, Docker

**Frontend:** TypeScript, HTML, CSS

**Machine Learning:** Scikit-learn, LightGBM, XGBoost, CatBoost

**MLOps & CI/CD**: MLFlow,  Github Actions

**Data wrangling:** Polars, Pandas, NumPy

**Optimsisation & Feature engineering:** Optuna, OpenFE

**Deployment:** HuggingFace Spaces