# Start API
`uv run fastapi dev main.py`

# Deployment Pattern (No Runtime MLflow Dependency)
1. Download artifacts before image build:
`python -m api.scripts.download_inference_artifacts --metadata-run-id <RUN_ID> --tracking-uri <MLFLOW_TRACKING_URI>`
2. Build Docker image (artifacts are copied into `src/api/artifacts`):
`docker build -t football-analytics-api .`
3. Run API container:
`docker run -p 8000:8000 -e APP_API_KEY=<YOUR_KEY> football-analytics-api`
