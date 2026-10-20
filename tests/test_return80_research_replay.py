"""Installed portable replay executes both costs and fails closed at input boundaries."""

import json
import runpy
import sys

import numpy as np
import pytest
from test_replay_inputs import cache_fixture

from tushare_entitlement_hub.return80_research import replay
from tushare_entitlement_hub.return80_research.cli import main
from tushare_entitlement_hub.strategy_discovery.replay_inputs import export_replay_inputs


def replay_fixture(root):
    """Build the eight-session public fixture, with scores and a portable closure."""
    panel, scores = cache_fixture(root / "cache")
    panel_root = root / "cache/p03-regime-alpha-20261018/panel"
    metadata = json.loads((panel_root / "panel.json").read_text())
    metadata["market"].append("index_open")
    (panel_root / "panel.json").write_text(json.dumps(metadata))
    np.save(panel_root / "market-index_open.npy", panel.market["index"])
    export_replay_inputs(root / "cache", root / "dataset")
    path = root / "scores.npy"
    np.save(path, scores)
    return (
        panel,
        scores,
        [
            "--dataset",
            str(root / "dataset"),
            "--scores",
            str(path),
            "--output",
            str(root / "accounts"),
            "--start",
            panel.calendar[0],
            "--scope",
            "original",
            "--capture-execution-evidence",
        ],
    )


def test_complete_two_cost_replay_from_installed_cli(tmp_path):
    """The real offline engine emits both accounts and successful arithmetic audits."""
    _, _, args = replay_fixture(tmp_path)
    known = tmp_path / "known.npy"
    np.save(known, np.ones((8, 12), dtype=bool))
    main(["replay", *args, "--known-mask", str(known)])
    reports = json.loads((tmp_path / "accounts/results.json").read_text())
    assert [item["cost"] for item in reports] == [1, 2]
    for name in ("baseline_cost1", "baseline_cost2"):
        assert (tmp_path / "accounts" / name / "daily.csv").exists()
        assert json.loads((tmp_path / "accounts" / name / "audit.json").read_text())


@pytest.mark.parametrize(
    "failure",
    [
        "capital",
        "scores-required",
        "existing",
        "shape",
        "inf",
        "known-shape",
        "known-dtype",
        "missing",
        "liquidity",
    ],
)
def test_replay_rejects_invalid_contracts(tmp_path, failure):
    """No malformed score/known mask, missing eligible prediction or invalid economics can run."""
    _, scores, args = replay_fixture(tmp_path)
    expected_error = ""
    if failure == "capital":
        args += ["--capital", "0"]
        expected_error = "initial_capital"
    elif failure == "scores-required":
        args = args[:2] + args[4:]
        expected_error = "explicit_scores"
    elif failure == "existing":
        (tmp_path / "accounts").mkdir()
        expected_error = "fresh_output"
    elif failure == "shape":
        np.save(tmp_path / "scores.npy", np.ones(3))
        expected_error = "score_axis"
    elif failure == "inf":
        scores[0, 0] = np.inf
        np.save(tmp_path / "scores.npy", scores)
        expected_error = "score_axis"
    elif failure.startswith("known-"):
        np.save(
            tmp_path / "known.npy",
            np.ones((8, 12) if failure == "known-dtype" else (1,), dtype=float),
        )
        args += ["--known-mask", str(tmp_path / "known.npy")]
        expected_error = "known_mask"
    elif failure == "missing":
        np.save(tmp_path / "scores.npy", np.full(scores.shape, np.nan))
        expected_error = "missing_scores"
    else:
        args += ["--minimum-signal-liquidity", "100"]
        expected_error = "requires_execution_rule"
    with pytest.raises(ValueError, match=expected_error):
        replay.main(args)


def test_replay_module_entry_requires_explicit_input(monkeypatch):
    """Running the module without arguments prints argparse usage instead of assuming a cache."""
    monkeypatch.setattr(sys, "argv", ["replay"])
    with pytest.warns(RuntimeWarning, match="found in sys.modules"), pytest.raises(SystemExit):
        runpy.run_module("tushare_entitlement_hub.return80_research.replay", run_name="__main__")


def test_account_rejects_selection_outside_scope(tmp_path, monkeypatch):
    """A forbidden engine selection fails before account files are published."""
    _, _, args = replay_fixture(tmp_path)
    scope = np.ones(12, dtype=bool)
    scope[0] = False
    monkeypatch.setattr(replay, "account_scope_mask", lambda *args: scope)
    monkeypatch.setattr(
        replay,
        "simulate_candidate_allocation",
        lambda *args, **kwargs: {"candidate_selections": [{"selected": np.array([0])}]},
    )
    with pytest.raises(ValueError, match="selection_outside_account_scope"):
        replay.main(args)
    assert not (tmp_path / "accounts/results.json").exists()
