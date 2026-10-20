"""Batch-size invariance and closed failures for native row-wise prediction."""

from types import SimpleNamespace

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.native_prediction import predict_native_batches


@pytest.mark.parametrize("chunk_rows", [1, 2, 10])
def test_native_batches_preserve_order_and_values(chunk_rows):
    """Scenario: several batch boundaries, including a short tail; expect identical scores."""
    features = np.arange(6).reshape(3, 2)
    model = SimpleNamespace(predict=lambda batch: batch.sum(axis=1))
    result = predict_native_batches(model, features, chunk_rows=chunk_rows)
    np.testing.assert_array_equal(result, [1., 5., 9.])
    assert result.dtype == np.float64


@pytest.mark.parametrize("features,chunk_rows", [
    (np.ones(2), 1), (np.ones((0, 2)), 1), (np.ones((2, 0)), 1),
    (np.ones((2, 2)), True), (np.ones((2, 2)), 1.5), (np.ones((2, 2)), 0),
])
def test_native_batches_reject_invalid_contract(features, chunk_rows):
    """Scenario: empty/bad shape or invalid chunk length; expect failure before prediction."""
    with pytest.raises(ValueError, match="contract"):
        predict_native_batches(None, features, chunk_rows=chunk_rows)


@pytest.mark.parametrize("prediction", [np.ones((2, 1)), np.array([np.nan, 1.])])
def test_native_batches_reject_invalid_scores(prediction):
    """Scenario: multioutput or nonfinite prediction; expect closed failure."""
    model = SimpleNamespace(predict=lambda batch: prediction)
    with pytest.raises(ValueError, match="scores"):
        predict_native_batches(model, np.ones((2, 1)), chunk_rows=2)
