from typing import Any
from pydantic import BaseModel

class InferenceResponse(BaseModel):
    prediction: Any
    probability: float | None = None
    timestamp: str
