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

## Pass Prediction Proxy Secrets
Set these environment variables on the backend host:
- `EXTERNAL_PASS_API_KEY`: API key used for `X-API-Key` to the external pass endpoint
- `EXTERNAL_PASS_API_URL` (optional): defaults to `https://yuch0001-football-api.hf.space/predict`
- `APP_CORS_ORIGINS` (optional): comma-separated origins, or `*`

The frontend should call `/api/v1/pass-prediction/predict` and must not include API keys.

# Deployment Pattern (No Runtime MLflow Dependency)
1. Download artifacts before image build:
`python -m api.scripts.download_inference_artifacts --tracking-uri <MLFLOW_TRACKING_URI> --output-dir src/api/artifacts --model-name "Final models_lightgbm" --model-alias production --final-models-experiment-id <FINAL_MODELS_EXPERIMENT_ID> --model-selection-experiment-id <MODEL_SELECTION_EXPERIMENT_ID>`
2. Build Docker image (artifacts are copied into `src/api/artifacts`):
`docker build -t football-analytics-api:latest -f Dockerfile .`
3. Run API container:
`docker run --rm -p 8000:7860 football-analytics-api:latest`

The artifact download script writes these files to `src/api/artifacts` by default:
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