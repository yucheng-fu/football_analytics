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
    # New argument to drive pre-allocation
    n_samples: int = field(default=0, kw_only=True)
    n_classes: int = field(default=2, kw_only=True)
    out_of_fold_predictions: Optional[np.ndarray] = field(default=None, init=False)


def __post_init__(self):
    if self.n_samples > 0:
        if self.n_classes > 2:
            # Multiclass: Keep all class probabilities (N x C)
            self.out_of_fold_predictions = np.zeros((self.n_samples, self.n_classes))
        else:
            # Binary: Optimize memory by storing only class 1 as a 1D array (N,)
            self.out_of_fold_predictions = np.zeros(self.n_samples)


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
