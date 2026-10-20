"""Portable matrices preserve the baseline and join extensions by dated stock identities."""

import csv
import json
from datetime import date, timedelta

import numpy as np
import pytest

from tushare_entitlement_hub.return80_research.dataset import (
    REQUIRED,
    append_features,
    import_training_bundle,
    load_dataset,
    map_predictions,
    save_dataset,
    valid_date,
    validate_arrays,
)
from tushare_entitlement_hub.strategy_discovery.replay_inputs import file_digest
from tushare_entitlement_hub.strategy_discovery.training_weights import daily_equal_training_weights


def arrays_fixture():
    """Return two mature candidates and one censored row with one feature and known weights."""
    calendar = [(date(2021, 10, 18) + timedelta(days=day)).strftime("%Y%m%d") for day in range(40)]
    days, securities = np.array([0, 1, 39]), np.array([0, 1, 0])
    labels = np.array([[0.05, 2, 2, 3], [-0.08, 2, 3, 3], [np.nan, np.nan, np.nan, 2]])
    weights, _ = daily_equal_training_weights(
        days[:2], labels[:2, 2], calendar, cutoff=calendar[-1], half_life_sessions=126
    )
    return dict(
        X=np.array([[1.0], [np.nan], [3.0]], dtype=np.float32),
        fit_indices=np.array([0, 1]),
        y=labels[:2, 0],
        sample_weight=weights,
        feature_names=np.array(["base"]),
        candidate_dates=np.asarray(calendar)[days],
        candidate_codes=np.array(["A", "B", "A"]),
        candidate_day_indices=days,
        candidate_security_indices=securities,
        common_eligible=np.ones(3, dtype=bool),
        labels=labels,
        calendar=np.asarray(calendar),
        security_codes=np.array(["A", "B"]),
    )


def block_fixture(root, rows):
    """Write a daily factor CSV and its hash manifest; rows may be reordered or incomplete."""
    path = root / "factor.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trade_date", "ts_code", "available_date", "factor"])
        writer.writerows(rows)
    spec = {
        "schema": "return80-feature-block/v1",
        "feature_names": ["factor"],
        "file": path.name,
        "sha256": file_digest(path),
        "source": "fixture.daily",
        "units": {"factor": "ratio"},
        "formula_version": "fixture/v1",
    }
    manifest = root / "block.json"
    manifest.write_text(json.dumps(spec))
    return manifest


def test_roundtrip_join_and_prediction_axis(tmp_path):
    """Reordered partial factors retain NaNs, targets and weights; scores map to dated grid."""
    source, extended = tmp_path / "data", tmp_path / "extended"
    arrays = arrays_fixture()
    save_dataset(source, arrays, {"source": "fixture"})
    block = block_fixture(
        tmp_path,
        [
            ["20211019", "B", "20211019", "2"],
            ["20211018", "A", "20211017", ""],
        ],
    )
    append_features(source, block, extended)
    loaded, manifest = load_dataset(extended)
    np.testing.assert_array_equal(loaded["X"][:, 0], arrays["X"][:, 0])
    np.testing.assert_array_equal(loaded["X"][:, 1], [np.nan, 2, np.nan])
    for key in set(REQUIRED) - {"X", "feature_names"}:
        np.testing.assert_array_equal(loaded[key], arrays[key])
    assert manifest["provenance"]["missing_rows"] == 1
    predictions = tmp_path / "predictions.npy"
    np.save(predictions, [0.1, 0.2, 0.3])
    output = tmp_path / "scores.npy"
    map_predictions(extended, predictions, output)
    scores = np.load(output)
    assert scores.shape == (40, 2)
    assert scores[0, 0] == 0.1 and scores[1, 1] == 0.2 and scores[39, 0] == 0.3
    assert np.isfinite(scores).sum() == 3
    with pytest.raises(ValueError, match="fresh_predictions"):
        map_predictions(extended, predictions, output)
    np.save(predictions, [0.1, 0.2])
    with pytest.raises(ValueError, match="invalid_candidate_predictions"):
        map_predictions(extended, predictions, tmp_path / "bad.npy")
    with pytest.raises(ValueError, match="fresh_training_output"):
        save_dataset(source, arrays, {})


@pytest.mark.parametrize(
    "damage", ["missing", "inf", "identity", "index", "targets", "weights", "date"]
)
def test_corrupt_training_arrays_fail_closed(damage):
    """Missing data, wrong identities, invalid labels/weights and impossible dates are refused."""
    arrays = arrays_fixture()
    if damage == "missing":
        del arrays["X"]
    elif damage == "inf":
        arrays["X"][0, 0] = np.inf
    elif damage == "identity":
        arrays["candidate_codes"][0] = "B"
    elif damage == "index":
        arrays["candidate_day_indices"][0] = -1
    elif damage == "targets":
        arrays["fit_indices"] = np.array([1])
    elif damage == "weights":
        arrays["sample_weight"][0] = 0
    else:
        arrays["calendar"][0] = "20211301"
    with pytest.raises(ValueError):
        validate_arrays(arrays)


@pytest.mark.parametrize(
    "rows",
    [
        [["20211018", "A", "20211019", "1"]],
        [["20211018", "A", "20211018", "1"], ["20211018", "A", "20211018", "1"]],
        [["20211018", "Z", "20211018", "1"]],
        [["20211018", "A", "20211018", "inf"]],
        [["20211018", "A", "20211018", "1e100"]],
    ],
)
def test_invalid_extensions_rejected(tmp_path, rows):
    """Future availability, duplicates, extra stocks, infinities and overflow cannot enter X."""
    source = tmp_path / "data"
    save_dataset(source, arrays_fixture(), {})
    with pytest.raises(ValueError):
        append_features(source, block_fixture(tmp_path, rows), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_import_delivered_bundle_verifies_hashes_and_protocol(tmp_path):
    """The original ZIP layout imports without reading any original machine paths."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    arrays = arrays_fixture()
    np.savez_compressed(bundle / "training.npz", **arrays)
    protocol = {
        "feature_names": ["base"],
        "candidate_rows": 3,
        "fit_rows": 2,
        "training_weights": {"half_life_sessions": 126},
    }
    (bundle / "original-protocol.json").write_text(json.dumps(protocol))
    (bundle / "SHA256SUMS").write_text(
        "".join(
            file_digest(bundle / name) + "  " + name + "\n"
            for name in ("training.npz", "original-protocol.json")
        )
    )
    import_training_bundle(bundle, tmp_path / "imported")
    loaded, _ = load_dataset(tmp_path / "imported")
    np.testing.assert_array_equal(loaded["X"], arrays["X"])
    (bundle / "original-protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="changed_bundle_file"):
        import_training_bundle(bundle, tmp_path / "rejected")


def test_manifest_and_block_tampering(tmp_path):
    """Modified files, duplicate feature names, missing CSV fields and schema changes fail."""
    source = tmp_path / "data"
    save_dataset(source, arrays_fixture(), {})
    block = block_fixture(tmp_path, [])
    spec = json.loads(block.read_text())
    spec["feature_names"] = ["base"]
    block.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="invalid_feature_block"):
        append_features(source, block, tmp_path / "bad")
    manifest = source / "dataset.json"
    spec = json.loads(manifest.read_text())
    spec["candidate_rows"] = 4
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="dataset_manifest_axis"):
        load_dataset(source)
    (source / "X.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed_dataset_file"):
        load_dataset(source)
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="invalid_dataset_manifest"):
        load_dataset(source)
    assert not valid_date("20240230")
