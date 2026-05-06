import os
from datetime import datetime, timezone
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader
from schemas.request import InferenceRequest
from schemas.response import InferenceResponse

router = APIRouter(prefix="/inference")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str | None = Depends(api_key_header)) -> str:
    load_dotenv()
    expected_key = os.getenv("APP_API_KEY")
    if not expected_key or api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: invalid API key",
        )
    return api_key

@router.get("/health", tags=["Inference"])
def health(request: Request) -> dict[str, Any]:
    model_loaded = getattr(request.app.state, "model", None) is not None
    return {"status": "ok", "model_loaded": model_loaded}


@router.post(
    "/predict",
    tags=["Inference"],
    response_model=InferenceResponse,
    responses={
        401: {"description": "Unauthorized: invalid API key"},
        422: {"description": "Validation Error"},
        500: {"description": "Inference failed"},
    },
)
def predict(
    payload: InferenceRequest,
    request: Request,
    _: str = Depends(verify_api_key),
) -> InferenceResponse:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    try:
        input_df = pd.DataFrame([payload.model_dump()])
        # prediction_raw = model.predict(input_df)

        # prediction = prediction_raw[0]
        # if isinstance(prediction, np.generic):
        #     prediction = prediction.item()

        # probability = None
        # if hasattr(model, "predict_proba"):
        #     proba_raw = model.predict_proba(input_df)
        #     if isinstance(proba_raw, np.ndarray):
        #         if proba_raw.ndim == 2 and proba_raw.shape[1] > 1:
        #             probability = float(proba_raw[0, 1])
        #         else:
        #             probability = float(proba_raw[0])

        return InferenceResponse(
            prediction="success",
            probability=0.97,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except mlflow.exceptions.MlflowException as exc:
        raise HTTPException(status_code=500, detail="Inference failed") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed") from exc
