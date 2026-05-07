import mlflow
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request

from api.schemas.request import InferenceRequest
from api.v1.routes.inference import verify_api_key
from utils.utils import prepare_inference_frame

router = APIRouter(prefix="/data")


@router.post(
    "/inference-frame",
    tags=["Data"],
    responses={
        401: {"description": "Unauthorized: invalid API key"},
        422: {"description": "Validation Error"},
        500: {"description": "Inference frame generation failed"},
    },
)
def inference_frame(
    payload: InferenceRequest,
    request: Request,
    _: str = Depends(verify_api_key),
) -> dict:
    bundle = getattr(request.app.state, "inference_bundle", None)
    if bundle is None:
        raise HTTPException(status_code=500, detail="Inference bundle is not loaded")

    fitted_column_transformer = bundle.get("fitted_column_transformer")
    row_wise_features = bundle.get("row_wise_features")
    column_wise_features = bundle.get("column_wise_features")
    best_features = bundle.get("best_features", bundle.get("selected_features"))
    categorical_mapping = bundle.get("categorical_mapping")

    try:
        input_df = pd.DataFrame([payload.model_dump(mode="json")])
        frame = prepare_inference_frame(
            X_pd=input_df,
            row_wise_features=row_wise_features,
            column_wise_features=column_wise_features,
            column_transformer=fitted_column_transformer,
            best_features=best_features,
            categorical_mapping=categorical_mapping,
        )
        return {"inference_frame": frame.to_dict(orient="records")}
    except mlflow.exceptions.MlflowException as exc:
        raise HTTPException(
            status_code=500, detail=f"Inference frame generation failed: {exc}"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Inference frame generation failed: {exc}"
        ) from exc
