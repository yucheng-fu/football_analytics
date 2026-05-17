import numpy as np
from dataclasses import dataclass, field
from pydantic import BaseModel
from typing import List, Any, Optional


@dataclass
class OuterCVResults:
    scores: List[float] = field(default_factory=list)
    params: List[dict] = field(default_factory=list)
    features: List[Any] = field(default_factory=list)
    parent_run_id: Optional[str] = None
    run_ids: List[str] = field(default_factory=list)
    experiment_ids: List[str] = field(default_factory=list)


class LGBMParams(BaseModel):
    colsample_bytree: Optional[float] = 1.0
    learning_rate: Optional[float] = 0.1
    num_leaves: Optional[int] = 31
    n_estimators: Optional[int] = 100
    reg_alpha: Optional[float] = 0.0
    reg_lambda: Optional[float] = 0.0
    subsample: Optional[float] = 1.0


class XGBoostParams(BaseModel):
    n_estimators: Optional[int] = 100
    max_depth: Optional[int] = 6
    grow_policy: Optional[str] = "depthwise"
    learning_rate: Optional[float] = 0.1
    subsample: Optional[float] = 1.0
    colsample_bytree: Optional[float] = 1.0
    reg_alpha: Optional[float] = 0.0
    reg_lambda: Optional[float] = 0.0


class CatBoostParams(BaseModel):
    iterations: Optional[int] = 100
    depth: Optional[int] = 6
    learning_rate: Optional[float] = 0.1
    reg_lambda: Optional[float] = 0.0
