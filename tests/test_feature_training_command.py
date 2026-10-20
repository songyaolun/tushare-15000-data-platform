"""Exercise the research command with tiny deterministic local arrays and real HGB."""

import hashlib
import json
import runpy
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")


@pytest.fixture
def training_spec(tmp_path):
    """Build a 40-day fixture with nine mature rows and two added event columns."""
    cache = tmp_path / "cache"
    base = cache / "stockops-return80-model-20261019"
    (base / "models").mkdir(parents=True)
    axis_root = cache / "stockops-utility-score-20261019"
    axis_root.mkdir()
    panel_root = cache / "p03-regime-alpha-20261018/panel"
    panel_root.mkdir(parents=True)
    label_root = cache / "stockops-joint-score-exit-20261019"
    label_root.mkdir()
    np.save(base / "features.npy", np.arange(40.0, dtype=np.float32)[:, None])
    np.savez(axis_root / "candidate-axis.npz", days=np.arange(40), securities=np.zeros(40, int))
    (panel_root / "panel.json").write_text(
        json.dumps(
            {"calendar": [(date(2021, 10, 18) + timedelta(days=x)).strftime("%Y%m%d")
                          for x in range(40)], "codes": ["000001.SZ"]}
        )
    )
    labels = np.column_stack(
        (np.arange(40.0) / 1000, np.full(40, 2), np.arange(40) + 2, np.full(40, 3))
    )
    np.save(label_root / "long-labels.npy", labels)
    (base / "models/hgb_leaves31_rounds1600.json").write_text(
        json.dumps(
            {
                "feature_names": ["base"],
                "parameters": {
                    "max_iter": 2,
                    "min_samples_leaf": 2,
                    "early_stopping": False,
                    "random_state": 1,
                },
            }
        )
    )
    baseline_path = tmp_path / "baseline.npy"
    event_path = tmp_path / "event.npy"
    sectors_path = tmp_path / "sectors.npy"
    np.save(baseline_path, np.zeros((40, 6), dtype=np.float32))
    np.save(event_path, np.column_stack((np.arange(40.0), np.arange(40.0) ** 2)))
    np.save(sectors_path, np.zeros(40, dtype=int))

    def block(path, names):
        """Return an artifact pin; a local NPY path maps to hash and ordered names."""
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "feature_names": names,
        }

    spec = {
        "cache_root": str(cache),
        "output_root": str(tmp_path / "output"),
        "cases": {"event": ["event"], "event_placebo": ["event"]},
        "candidate_rows": 40,
        "fit_rows": 9,
        "source_revision": "offline-fixture",
        "sklearn_version": sklearn.__version__,
        "baseline_additional": block(baseline_path, [f"b{x}" for x in range(6)]),
        "groups": {"event": block(event_path, ["age", "reaction"])},
        "candidate_sectors": block(sectors_path, []),
        "pinned_inputs": [block(base / "features.npy", ["base"])],
    }
    return tmp_path / "spec.json", spec


@pytest.mark.parametrize("case", ["event", "event_placebo"])
def test_real_hgb_command_exports_verified_complete_axis(training_spec, monkeypatch, case):
    """A real tiny model exports exact portable predictions and the entire date/security axis."""
    path, spec = training_spec
    path.write_text(json.dumps(spec))
    monkeypatch.setattr(sys, "argv", ["train_feature_case", "--spec", str(path), "--case", case])
    runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/research/train_feature_case.py"),
        run_name="__main__",
    )
    root = Path(spec["output_root"]) / case
    audit = json.loads((root / "audit.json").read_text())
    assert audit["fit_rows"] == 9 and audit["columns"] == 9 and audit["trees"] == 2
    assert audit["portable_prediction_max_error"] is None
    assert audit["portable_validation"]["maximum_absolute_error"] == 0
    assert audit["portable_validation"]["full_portable_verified"] is False
    np.testing.assert_array_equal(
        np.load(root / "scores.npy")[:, 0], np.load(root / "candidate-prediction.npy")
    )
    if case == "event_placebo":
        assert json.loads((root / "protocol.json").read_text())["placebo"]["rows"] == 40


@pytest.mark.parametrize(
    "failure", ["case", "existing", "pin", "version", "fit", "sectors", "duplicate"]
)
def test_training_command_rejects_changed_contract(training_spec, monkeypatch, failure):
    """Unregistered cases, changed pins/versions/axes or duplicate names fail before fitting."""
    path, spec = training_spec
    case = "event_placebo" if failure == "sectors" else "event"
    if failure == "case":
        case = "unregistered"
    elif failure == "existing":
        (Path(spec["output_root"]) / case).mkdir(parents=True)
    elif failure == "pin":
        spec["pinned_inputs"][0]["sha256"] = "wrong"
    elif failure == "version":
        spec["sklearn_version"] = "wrong"
    elif failure == "fit":
        spec["fit_rows"] = 10
    elif failure == "sectors":
        spec["candidate_sectors"]["sha256"] = "wrong"
    else:
        spec["groups"]["event"]["feature_names"][0] = "base"
    path.write_text(json.dumps(spec))
    monkeypatch.setattr(sys, "argv", ["train_feature_case", "--spec", str(path), "--case", case])
    with pytest.raises(ValueError):
        runpy.run_path(
            str(Path(__file__).parents[1] / "scripts/research/train_feature_case.py"),
            run_name="__main__",
        )


def test_training_progress_preserves_portable_model_and_predictions(
    training_spec, monkeypatch, capsys,
):
    """Official verbosity and phase reporting preserve every portable tree and prediction."""
    path, spec = training_spec
    exports = []
    for quiet in (False, True):
        spec["output_root"] = str(path.parent / ("quiet" if quiet else "progress"))
        path.write_text(json.dumps(spec))
        arguments = ["train", "--spec", str(path), "--case", "event"]
        if quiet:
            arguments.append("--quiet-progress")
        monkeypatch.setattr(sys, "argv", arguments)
        runpy.run_path(str(Path(__file__).parents[1] / "scripts/research/train_feature_case.py"),
                       run_name="__main__")
        output = capsys.readouterr().out
        records = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
        events = [record["event"] for record in records if "event" in record]
        assert events == ([] if quiet else [
            "fit_start", "fit_done", "native_predict_start", "native_predict_done",
            "native_checkpoint_start", "native_checkpoint_done",
            "portable_verify_start", "portable_verify_done",
        ])
        root = Path(spec["output_root"]) / "event"
        exports.append(((root / "model-portable.json").read_bytes(),
                        np.load(root / "candidate-prediction.npy")))
    assert exports[0][0] == exports[1][0]
    np.testing.assert_array_equal(exports[0][1], exports[1][1])


def test_native_checkpoint_survives_portable_validation_failure(training_spec, monkeypatch):
    """Export-check failure preserves hashed native scores/model without claiming certification."""
    path, spec = training_spec
    path.write_text(json.dumps(spec))

    def fail_validation(*args, **kwargs):
        """Simulate export verification failure after native persistence; never refit."""
        raise RuntimeError("portable_mismatch")

    monkeypatch.setattr(
        "tushare_entitlement_hub.strategy_discovery.portable_validation."
        "validate_portable_predictions", fail_validation,
    )
    monkeypatch.setattr(sys, "argv", ["train", "--spec", str(path), "--case", "event"])
    with pytest.raises(RuntimeError, match="portable_mismatch"):
        runpy.run_path(str(Path(__file__).parents[1] / "scripts/research/train_feature_case.py"),
                       run_name="__main__")
    root = Path(spec["output_root"]) / "event"
    checkpoint = json.loads((root / "native-checkpoint.json").read_text())
    assert checkpoint["status"] == "native_ready" and not (root / "audit.json").exists()
    for name, expected in checkpoint["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    np.testing.assert_array_equal(np.load(root / "scores.npy")[:, 0],
                                  np.load(root / "candidate-prediction.npy"))


def test_bounded_and_full_modes_preserve_native_scores(training_spec, monkeypatch):
    """Changing only export-check scope preserves the model and all native account scores."""
    path, spec = training_spec
    outputs = []
    for mode in ("bounded", "full"):
        spec["output_root"] = str(path.parent / mode)
        path.write_text(json.dumps(spec))
        monkeypatch.setattr(sys, "argv", ["train", "--spec", str(path), "--case", "event",
                                         "--portable-validation", mode])
        runpy.run_path(str(Path(__file__).parents[1] / "scripts/research/train_feature_case.py"),
                       run_name="__main__")
        root = Path(spec["output_root"]) / "event"
        audit = json.loads((root / "audit.json").read_text())
        assert audit["portable_validation"]["mode"] == mode
        assert audit["portable_validation"]["full_portable_verified"] is (mode == "full")
        outputs.append(((root / "model-portable.json").read_bytes(),
                        (root / "scores.npy").read_bytes()))
    assert outputs[0] == outputs[1]
