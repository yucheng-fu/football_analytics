import os
from datetime import datetime, timezone
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader
from api.schemas.request import InferenceRequest
from api.schemas.response import InferenceResponse
from utils.utils import prepare_inference_frame

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
    bundle = getattr(request.app.state, "inference_bundle", None)
    model_loaded = bundle is not None and bundle.get("model") is not None
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
    """_summary_

    Args:
        payload (InferenceRequest): _description_
        request (Request): _description_
        _ (str, optional): _description_. Defaults to Depends(verify_api_key).

    Raises:
        HTTPException: _description_
        HTTPException: _description_
        HTTPException: _description_
        HTTPException: _description_
        HTTPException: _description_

    Returns:
        InferenceResponse: _description_
    """
    bundle = getattr(request.app.state, "inference_bundle", None)
    if bundle is None or bundle.get("model") is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    model = bundle["model"]
    fitted_column_transformer = bundle.get("fitted_column_transformer")
    row_wise_features = bundle.get("row_wise_features")
    column_wise_features = bundle.get("column_wise_features")
    best_features = bundle.get("best_features", bundle.get("selected_features"))
    categorical_mapping = bundle.get("categorical_mapping")

    try:
        input_df = pd.DataFrame([payload.model_dump(mode="json")])
        model_input = prepare_inference_frame(
            X_pd=input_df,
            row_wise_features=row_wise_features,
            column_wise_features=column_wise_features,
            column_transformer=fitted_column_transformer,
            best_features=best_features,
            categorical_mapping=categorical_mapping,
        )
        if not hasattr(model, "predict_proba"):
            raise HTTPException(
                status_code=500, detail="Loaded model does not support predict_proba"
            )

        proba_raw = model.predict_proba(model_input)
        proba_array = np.asarray(proba_raw)
        if proba_array.ndim != 2:
            raise HTTPException(status_code=500, detail="Invalid predict_proba output")

        class_idx = int(np.argmax(proba_array[0]))
        probability = float(proba_array[0, class_idx])
        classes = getattr(model, "classes_", None)
        prediction = classes[class_idx] if classes is not None else class_idx

        return InferenceResponse(
            prediction=prediction,
            probability=probability,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except mlflow.exceptions.MlflowException as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
