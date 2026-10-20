"""Ordered native estimator prediction with an explicit cache-sized row bound."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def predict_native_batches(
    model: Any, features: NDArray[Any], *, chunk_rows: int,
) -> NDArray[np.float64]:
    """Predict independent rows in order, rejecting malformed or nonfinite scores.

    Example: a row-sum estimator on [[1,2],[3,4],[5,6]], chunk_rows=2 returns
    [3.,7.,11.] via native batches of two and one rows. Only use for estimators
    whose prediction is row-independent, such as frozen HGB; verify parity with
    the original complete native prediction before adopting a chunk size.
    """
    if (features.ndim != 2 or not len(features) or not features.shape[1]
            or isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int)
            or chunk_rows < 1):
        raise ValueError("invalid native prediction batch contract")
    parts = []
    for start in range(0, len(features), chunk_rows):
        batch = features[start:start + chunk_rows]
        prediction = np.asarray(model.predict(batch), dtype=np.float64)
        if prediction.shape != (len(batch),) or not np.isfinite(prediction).all():
            raise ValueError("invalid native prediction scores")
        parts.append(prediction)
    return np.concatenate(parts)
