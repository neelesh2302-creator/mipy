"""Settings shared by every imputer.

An :class:`ImputeContext` carries everything an imputer may need besides
the matrix itself: the seed, the patient and time of each row, the
predictor sets, and the tuning constants. It is a plain dataclass, so it
can be sent to worker processes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np


@dataclass
class ImputeContext:
    seed: int = 2026
    patient: Optional[np.ndarray] = None      # patient id of each row
    time: Optional[np.ndarray] = None         # visit / sampling point of each row
    schema: Optional[dict] = None             # {column: [predictor columns]}
    mice_iter: int = 5
    m: int = 5                                # completions for MI methods
    knn_k: int = 15
    max_predictors: int = 25
    min_cor: float = 0.10
    gain_epochs: int = 1000                   # hyperimpute's default
    hyper_inner_iter: int = 3
    gamma: float = 1.0                        # MNAR sensitivity parameter (MICE-LongMNAR)
    mi_holdout: float = 0.10                  # share of observed cells hidden per draw (MI wrappers)
    extra: dict = field(default_factory=dict)

    def with_(self, **kw) -> "ImputeContext":
        return replace(self, **kw)

    def rows_by_patient(self) -> np.ndarray:
        """Patient labels; one patient per row when none were given."""
        if self.patient is None:
            raise ValueError("this method needs ctx.patient (the patient of each row)")
        return np.asarray(self.patient)
