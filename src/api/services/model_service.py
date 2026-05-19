from datetime import datetime, timezone
from typing import Any

import numpy as np

from api.schemas.request import InferenceRequest
from api.schemas.response import InferenceResponse
from api.services.inference_frame_service import InferenceFrameService


class ModelService:
    """Service for model-backed inference operations."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        """Initialize the model service.

        Args:
            bundle (dict[str, Any]): Loaded inference bundle.
        """
        self.bundle = bundle
        self.model = bundle.get("model")
        self.fitted_column_transformer = bundle.get("fitted_column_transformer")
        self.row_wise_features = bundle.get("row_wise_features")
        self.column_wise_features = bundle.get("column_wise_features")
        self.best_features = bundle.get("best_features", bundle.get("selected_features"))
        self.categorical_mapping = bundle.get("categorical_mapping")
        self.inference_frame_service = InferenceFrameService(bundle)

    def is_model_available(self) -> bool:
        """Check if model is loaded and available.

        Returns:
            bool: True if model is available.
        """
        return self.model is not None

    def predict(self, payload: InferenceRequest) -> InferenceResponse:
        """Run prediction for a single inference payload.

        Args:
            payload (InferenceRequest): Inference payload.

        Raises:
            ValueError: If model does not support required prediction interface.
            ValueError: If prediction output shape is invalid.

        Returns:
            InferenceResponse: Prediction response payload.
        """
        model_input = self.inference_frame_service.build_from_payload(payload)
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("Loaded model does not support predict_proba")

        proba_raw = self.model.predict_proba(model_input)
        proba_array = np.asarray(proba_raw)
        if proba_array.ndim != 2:
            raise ValueError("Invalid predict_proba output")

        class_idx = int(np.argmax(proba_array[0]))
        probability = float(proba_array[0, class_idx])
        classes = getattr(self.model, "classes_", None)
        prediction = classes[class_idx] if classes is not None else class_idx

        return InferenceResponse(
            prediction=prediction,
            probability=probability,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=type(self.model).__name__,
        )
