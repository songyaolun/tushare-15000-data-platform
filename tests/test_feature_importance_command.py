"""Offline CLI fixtures use an in-memory fake estimator, never real training."""

import hashlib
import json
import pickle
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/research/analyze_feature_importance.py"


@pytest.fixture
def protocol_fixture(tmp_path, monkeypatch):
    """Create four synthetic rows and a fake predictor; output protocol/path pair."""
    features = np.array([[1., 2.], [2., 1.], [3., np.nan], [4., np.nan]])
    np.save(tmp_path / "features.npy", features)
    np.savez(tmp_path / "axis.npz", days=np.array([0, 0, 1, 1]),
             securities=np.array([0, 1, 0, 1]), common=np.ones(4, dtype=bool))
    np.save(tmp_path / "sectors.npy", np.array([0, 1, 0, 1]))
    np.save(tmp_path / "labels.npy", np.array([[3, 1, 1, 3]] * 4, dtype=float))
    np.save(tmp_path / "prediction.npy", np.nansum(features, axis=1))
    (tmp_path / "panel.json").write_text(json.dumps({"calendar": ["20260101", "20260102"]}))
    (tmp_path / "model.pickle").write_bytes(b"fake-estimator-placeholder")
    nodes = np.array([(0, 0, 2.)],
                     dtype=[("is_leaf", "i4"), ("feature_idx", "i4"), ("gain", "f8")])
    model = SimpleNamespace(_predictors=[[SimpleNamespace(nodes=nodes)]],
                            predict=lambda matrix: np.nansum(matrix, axis=1))
    monkeypatch.setattr(pickle, "load", lambda stream: model)
    protocol = {"output": str(tmp_path / "output"), "pinned_inputs": [],
                "axis": str(tmp_path / "axis.npz"), "panel": str(tmp_path / "panel.json"),
                "feature_blocks": [str(tmp_path / "features.npy")],
                "sectors": str(tmp_path / "sectors.npy"),
                "labels": str(tmp_path / "labels.npy"), "dates": ["20260101", "20260102"],
                "signal_cutoff": "20260102", "exit_cutoff": "20260102", "rows": 4,
                "feature_names": ["first", "second"], "groups": {"both": [0, 1]},
                "models": {"fixture": {"model": str(tmp_path / "model.pickle"),
                                       "predictions": str(tmp_path / "prediction.npy")}},
                "seeds": [1, 2], "threads": 1}
    for path in tmp_path.iterdir():
        protocol["pinned_inputs"].append({"path": str(path),
                                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return protocol, tmp_path / "protocol.json"


def execute_fixture(protocol, path, monkeypatch):
    """Write one fixture protocol and run CLI main; example valid input yields receipt."""
    path.write_text(json.dumps(protocol))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--protocol", str(path)])
    runpy.run_path(str(SCRIPT), run_name="__main__")


def test_command_completes_bounded_permutations(protocol_fixture, monkeypatch):
    """Scenario: two columns plus one group and two seeds; expect six durable records."""
    protocol, path = protocol_fixture
    execute_fixture(protocol, path, monkeypatch)
    output = Path(protocol["output"])
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["completed"] == receipt["total"] == 6
    records = [json.loads(line) for line in (output / "fixture-permutation.jsonl")
               .read_text().splitlines()]
    assert len(records) == 6
    assert all(record["permutation"]["rows"] == 4 for record in records)
    assert len(receipt["artifacts"]) == 4
    with pytest.raises(FileExistsError):
        execute_fixture(protocol, path, monkeypatch)


@pytest.mark.parametrize("failure", ["hash", "shape", "infinity", "label", "parity"])
def test_command_fails_closed(protocol_fixture, monkeypatch, failure):
    """Scenario: changed bytes, dimensions, numeric inputs or scores; expect closed failure."""
    protocol, path = protocol_fixture
    if failure == "hash":
        protocol["pinned_inputs"][0]["sha256"] = "bad"
    elif failure == "shape":
        protocol["rows"] = 5
    else:
        protocol["pinned_inputs"] = []
        if failure == "infinity":
            np.save(Path(protocol["feature_blocks"][0]), np.full((4, 2), np.inf))
        elif failure == "label":
            np.save(Path(protocol["labels"]), np.array([[np.nan, 1, 1, 3]] * 4))
        else:
            np.save(Path(protocol["models"]["fixture"]["predictions"]), np.zeros(4))
    with pytest.raises(ValueError):
        execute_fixture(protocol, path, monkeypatch)


def test_command_import_does_not_run_main():
    """Scenario: importing the script for tooling; expect definitions and no execution."""
    namespace = runpy.run_path(str(SCRIPT))
    assert callable(namespace["main"]) and callable(namespace["digest"])


def test_command_resumes_completed_records_without_repredicting(protocol_fixture, monkeypatch):
    """Scenario: six completed records are resumed; expect six reused and exact records."""
    protocol, path = protocol_fixture
    execute_fixture(protocol, path, monkeypatch)
    original = Path(protocol["output"]) / "fixture-permutation.jsonl"
    protocol["output"] += "-resumed"
    protocol["resume_records"] = [str(original)]
    execute_fixture(protocol, path, monkeypatch)
    receipt = json.loads((Path(protocol["output"]) / "receipt.json").read_text())
    assert receipt["reused"] == receipt["completed"] == 6
    assert (Path(protocol["output"]) / original.name).read_bytes() == original.read_bytes()


@pytest.mark.parametrize("failure", ["duplicate", "key", "columns", "rows"])
def test_command_rejects_invalid_resume(protocol_fixture, monkeypatch, failure):
    """Scenario: malformed or duplicate resume key; expect failure before model prediction."""
    protocol, path = protocol_fixture
    record = {"model": "fixture", "kind": "feature", "name": "first", "seed": 1,
              "columns": [0], "permutation": {"rows": 4}}
    if failure == "key":
        record["seed"] = 99
    elif failure == "columns":
        record["columns"] = [1]
    elif failure == "rows":
        record["permutation"]["rows"] = 1
    resume = path.parent / "resume.jsonl"
    resume.write_text((json.dumps(record) + "\n") * (2 if failure == "duplicate" else 1))
    protocol["resume_records"] = [str(resume)]
    with pytest.raises(ValueError, match="resumed record"):
        execute_fixture(protocol, path, monkeypatch)


def test_command_groups_only_includes_daily_strata_evidence(protocol_fixture, monkeypatch):
    """Scenario: expanded validation uses groups only; expect two records and daily audits."""
    protocol, path = protocol_fixture
    protocol.update(groups_only=True, audit_by_day=True)
    execute_fixture(protocol, path, monkeypatch)
    records = [json.loads(line) for line in (Path(protocol["output"])
               / "fixture-permutation.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert all(row["kind"] == "group" for row in records)
    assert all(len(row["strata_by_day"]) == 2 for row in records)
