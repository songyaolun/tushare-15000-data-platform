"""Offline training resource gates and native artifact contracts."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_return80_research_dataset import arrays_fixture

from tushare_entitlement_hub.return80_research import training
from tushare_entitlement_hub.return80_research.cli import main
from tushare_entitlement_hub.return80_research.dataset import save_dataset


def test_cgroup_quotas_include_v1_v2_and_unlimited(tmp_path):
    """Nested quotas constrain the host; unlimited and malformed files contribute nothing."""
    (tmp_path / "cpu.max").write_text("200000 100000")
    directory = tmp_path / "cpu"
    directory.mkdir()
    (directory / "cpu.cfs_quota_us").write_text("100000")
    (directory / "cpu.cfs_period_us").write_text("100000")
    assert sorted(training.cpu_quotas(tmp_path)) == [1.0, 2.0]
    (tmp_path / "cpu.max").write_text("max 100000")
    (directory / "cpu.cfs_quota_us").write_text("-1")
    assert training.cpu_quotas(tmp_path) == []
    (tmp_path / "cpu.max").write_text("invalid")
    assert training.cpu_quotas(tmp_path) == []


def test_inspection_does_not_claim_gpu_support(monkeypatch):
    """GPU inventory is recorded while sklearn support remains explicitly false."""
    monkeypatch.setattr(
        training.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="fixture GPU, 8192 MiB"),
    )
    result = training.inspect_resources()
    assert result["cpu_limit"] >= 1 and result["available_memory_bytes"] > 0
    assert result["gpu_inventory"] == ["fixture GPU, 8192 MiB"]
    assert result["algorithm_gpu_supported"] is False


def test_missing_gpu_is_supported(monkeypatch):
    """A CPU-only machine has a valid resource report without inventing a GPU."""

    def unavailable(*args, **kwargs):
        """Simulate missing nvidia-smi; no network or actual GPU is needed."""
        raise FileNotFoundError

    monkeypatch.setattr(training.subprocess, "run", unavailable)
    assert training.inspect_resources()["gpu_inventory"] == []


def test_thread_probe_uses_actual_rows_and_checks_equal_predictions():
    """Two bounded probes use the same features including NaN and report selected threads."""
    x = np.arange(512.0, dtype=np.float32).reshape(256, 2)
    x[2, 1] = np.nan
    resource = {"cpu_limit": 2, "cpu_busy_percent": 0}
    selected, timings = training.select_threads(
        x,
        np.arange(256.0) / 256,
        np.ones(256),
        {"max_iter": 2, "min_samples_leaf": 5, "early_stopping": False, "random_state": 1},
        resource,
        2,
    )
    assert selected in (1, 2) and [item["threads"] for item in timings] == [1, 2]
    assert all(item["rows"] == 256 for item in timings)
    with pytest.raises(ValueError, match="thread_ceiling"):
        training.select_threads(x, np.ones(256), np.ones(256), {}, resource, 0)


def test_cli_native_training_and_resource_failures(tmp_path, monkeypatch):
    """An explicit one-round fixture saves native scores/metadata, and guards reject reuse."""
    data = tmp_path / "dataset"
    save_dataset(data, arrays_fixture(), {})
    resources = training.inspect_resources()
    resources.update(cpu_limit=1, cpu_busy_percent=0)
    monkeypatch.setattr(training, "inspect_resources", lambda: resources)
    output = tmp_path / "run"
    main(["train", "--dataset", str(data), "--output", str(output), "--rounds", "1"])
    complete = json.loads((output / "complete.json").read_text())
    assert complete["status"] == "native_ready"
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["financial_holdout"] is False
    assert protocol["parameters"]["early_stopping"] is False
    assert protocol["half_life_sessions"] == 126
    scores = np.load(output / "scores.npy")
    prediction = np.load(output / "candidate-prediction.npy")
    assert np.isfinite(scores).sum() == len(prediction) == 3
    with pytest.raises(ValueError, match="invalid_training"):
        training.train(data, output, rounds=1)
    with pytest.raises(ValueError, match="invalid_training"):
        training.train(data, tmp_path / "zero", rounds=0)
    resources["available_memory_bytes"] = 1
    with pytest.raises(ValueError, match="insufficient_training_memory"):
        training.train(data, tmp_path / "memory", rounds=1)
