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
`python -m api.scripts.download_inference_artifacts --metadata-run-id <RUN_ID> --tracking-uri <MLFLOW_TRACKING_URI>`
2. Build Docker image (artifacts are copied into `src/api/artifacts`):
`docker build -t football-analytics-api .`
3. Run API container:
`docker run -p 8000:8000 -e APP_API_KEY=<YOUR_KEY> football-analytics-api`
