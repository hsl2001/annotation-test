"""Model interface.

The concrete model lives outside this module.  The pipeline only depends on
this protocol so that any implementation can be injected through
``pipeline.run(..., model_factory=...)``.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class ModelProtocol(Protocol):
    label_names: list[str]
    thresholds: NDArray[np.float32] | None

    def fit(
        self,
        features: NDArray[np.float32],
        labels: dict[str, NDArray[np.int64]],
        motif_mask: NDArray[np.bool_] | None = None,
    ) -> ModelProtocol: ...

    def predict(self, features: NDArray[np.float32]) -> NDArray[np.float32]: ...

    def save(self, path: str) -> None: ...
