from contextlib import asynccontextmanager

import mlflow.pyfunc
import uvicorn
from fastapi import FastAPI

from api.v1.router import api_router

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     app.state.model = mlflow.pyfunc.load_model("./models")
#     yield


app = FastAPI(
    title="Football Analytics Inference API",
    description="Production-ready inference API for local MLflow model artifacts.",
    version="1.0.0",
    # lifespan=lifespan,
)

app.include_router(api_router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port="8000")