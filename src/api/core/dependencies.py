from fastapi import HTTPException, Request

from api.services.inference_frame_service import InferenceFrameService
from api.services.model_service import ModelService


def get_model_service(request: Request) -> ModelService:
    """Resolve model service from app state.

    Args:
        request (Request): FastAPI request object.

    Raises:
        HTTPException: If model service is not initialized.

    Returns:
        ModelService: Initialized model service.
    """
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        raise HTTPException(status_code=500, detail="Model service unavailable")
    return service


def get_inference_frame_service(request: Request) -> InferenceFrameService:
    """Resolve inference frame service from app state.

    Args:
        request (Request): FastAPI request object.

    Raises:
        HTTPException: If inference frame service is not initialized.

    Returns:
        InferenceFrameService: Initialized inference frame service.
    """
    service = getattr(request.app.state, "inference_frame_service", None)
    if service is None:
        raise HTTPException(status_code=500, detail="Inference frame service unavailable")
    return service
