"""Exercise the research command with tiny deterministic local arrays and real HGB."""

import hashlib
import json
import runpy
import sys
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
            {"calendar": ["20211018"] + [str(x) for x in range(1, 40)], "codes": ["000001.SZ"]}
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
    assert audit["portable_prediction_max_error"] == 0
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
