---
title: Football Api
emoji: 📊
colorFrom: indigo
colorTo: pink
sdk: docker
pinned: false
short_description: FastAPI endpoints
---

# Start API
`uv run fastapi dev main.py`

# Deployment Pattern (No Runtime MLflow Dependency)
1. Download artifacts before image build:
`python -m api.scripts.download_inference_artifacts --tracking-uri <MLFLOW_TRACKING_URI> --model-type <MODEL_TYPE> --model-alias production --final-models-experiment-id <FINAL_MODELS_EXPERIMENT_ID> --model-selection-experiment-id <MODEL_SELECTION_EXPERIMENT_ID>`
2. Build Docker image (artifacts are copied into `src/api/artifacts/<MODEL_TYPE>`):
`docker build -t football-analytics-api:latest -f Dockerfile .`
3. Run API container:
`docker run --rm -p 8000:7860 football-analytics-api:latest`

The artifact download script writes these files to `src/api/artifacts/<model_type>` by default:
- `model/`
- `row_wise_features.pkl`
- `column_wise_features.pkl`
- `fitted_column_transformer.pkl`
- `params.json`
- `best_params.json`
- `best_features.json`
- `selected_features.json`
- `categorical_mapping.json`
- `manifest.json`
