"""The replay CLI forwards one explicit candidate limit through both selection stages."""

import csv
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from test_discovery_candidate_allocation import allocation_fixture

from tushare_entitlement_hub.strategy_discovery.contracts import DEFAULT_POLICY
from tushare_entitlement_hub.strategy_discovery.signal_redesign import top_candidates


@pytest.mark.parametrize("count", [None, 20, 30])
@pytest.mark.parametrize("opening_cash_only", [False, True])
def test_cli_candidate_limit_reaches_sector_selection_and_allocation(
    tmp_path, monkeypatch, count, opening_cash_only, capsys,
):
    """Candidate counts and both cash modes reach allocation and the frozen protocol."""
    script_path = Path(__file__).parents[1] / "scripts/replay_return80_ablation.py"
    namespace = runpy.run_path(str(script_path))
    main = namespace["main"]
    panel, definition = allocation_fixture()
    panel.market["breadth"] = np.ones(8)
    context = (panel, None, DEFAULT_POLICY, definition, [], np.ones(8), None)
    for directory in ("stockops-candidate-allocation-v2-20261019", "stockops-sector-cap-20261019"):
        path = tmp_path / directory
        path.mkdir()
        (path / "run.py").write_text("# synthetic cache identity\n")
    scores_path = tmp_path / "scores.npy"
    np.save(scores_path, definition.scores)
    expected = 10 if count is None else count
    selected = top_candidates(definition.entry_mask, definition.scores, expected)
    sector_selection = Mock(return_value=selected)
    cached_loader = Mock(side_effect=[{"context": Mock(return_value=context)},
                                      {"sector_capped_mask": sector_selection}])
    monkeypatch.setitem(main.__globals__, "runpy", SimpleNamespace(run_path=cached_loader))
    liquidity_filter = Mock(return_value=(definition.entry_mask, [{"day": 0, "security": 0}]))
    monkeypatch.setitem(main.__globals__, "filter_signal_liquidity", liquidity_filter)
    allocation = Mock(side_effect=RuntimeError("allocation_boundary_reached"))
    monkeypatch.setitem(main.__globals__, "simulate_candidate_allocation", allocation)
    output = tmp_path / "output"
    arguments = ["replay", "--cache-root", str(tmp_path), "--output", str(output),
                 "--scores", str(scores_path), "--scope", "original",
                 "--maximum-daily-participation", ".01", "--minimum-signal-liquidity", "100"]
    if count is not None:
        arguments.extend(["--candidate-count", str(count)])
    if opening_cash_only:
        arguments.append("--opening-cash-only")
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(RuntimeError, match="allocation_boundary_reached"):
        main()
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["event"] for row in records if "event" in row] == [
        "replay_load_start", "replay_load_done", "account_replay_start", "account_replay_failed",
    ]
    assert sector_selection.call_count == 2
    for call in sector_selection.call_args_list:
        assert call.args[3] == 1
        assert call.kwargs["count"] == expected
    assert allocation.call_args.kwargs["count"] == expected
    assert allocation.call_args.kwargs["opening_cash_only"] is opening_cash_only
    assert int(allocation.call_args.args[3].entry_mask[0].sum()) == min(expected, 12)
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["execution_parameters"]["candidate_count"] == expected
    assert protocol["opening_cash_only"] is opening_cash_only
    assert f"Top{expected};" in protocol["signal_liquidity_placement"]
    with (output / "liquidity_candidates.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row[f"original_top{expected}"] == "True"
    assert row[f"filtered_top{expected}"] == "True"
    if expected != 10:
        assert "original_top10" not in row and "filtered_top10" not in row
