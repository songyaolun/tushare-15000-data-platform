"""Original feature, candidate and independent-label reconstruction contracts."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_discovery_candidate_allocation import allocation_fixture
from test_strategy_optimization import synthetic_rows

from tushare_entitlement_hub.return80_research import rebuild
from tushare_entitlement_hub.strategy_optimization.features import compute_features


def envelope(path, api, rows, parameters=None, fields=None):
    """Write a public source response; empty ST rows still retain explicit collected date."""
    fields = fields or list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "api_name": api,
                "parameters": parameters or {},
                "response": {
                    "code": 0,
                    "data": {
                        "fields": fields,
                        "items": [[row.get(key) for key in fields] for row in rows],
                    },
                },
            }
        )
    )


def fixture(tmp_path, count=150):
    """Produce one real technical history and original-unit supplemental inputs."""
    calendar, rows = synthetic_rows(count)
    source = tmp_path / "source"
    source.mkdir()
    for row in rows:
        row["amount"] = 100000.0
    features = compute_features(rows, calendar)
    features["sector_index"] = np.zeros(count)
    panel = SimpleNamespace(
        calendar=calendar,
        codes=["000001.SZ"],
        values={key: value[:, None] for key, value in features.items()},
        market={},
        signals={},
    )
    panel.values.update(beta=np.ones((count, 1)), capitalization=np.ones((count, 1)))
    np.savez(source / "000001.SZ.npz", **features)
    roots = [tmp_path / "raw"]
    for api, names in rebuild.SUPPLEMENT_FIELDS.items():
        records = [
            {"ts_code": "000001.SZ", "trade_date": day, **dict.fromkeys(names, 1.0)}
            for day in calendar
        ]
        if api != "stock_st":
            envelope(roots[0] / api / "all.json", api, records)
        else:
            for day in calendar:
                envelope(
                    roots[0] / api / (day + ".json"),
                    api,
                    [],
                    {"trade_date": day},
                    ["ts_code", "trade_date"],
                )
    benchmark = tmp_path / "benchmark.json"
    envelope(benchmark, "index_daily", [dict(row, ts_code="000300.SH") for row in rows])
    return panel, source, roots, benchmark


def test_supplemental_missing_dates_and_duplicates(tmp_path):
    """Missing ST partition is NaN; explicit empty is zero; conflicting rows reject."""
    panel, _, roots, _ = fixture(tmp_path, 3)
    (roots[0] / "stock_st" / (panel.calendar[1] + ".json")).unlink()
    arrays = rebuild.supplemental_arrays(panel, roots)
    np.testing.assert_array_equal(arrays["is_st"][:, 0], [0, np.nan, 0])
    envelope(
        roots[0] / "daily" / "duplicate.json",
        "daily",
        [{"ts_code": "000001.SZ", "trade_date": panel.calendar[0], "pre_close": 2}],
    )
    with pytest.raises(ValueError, match="conflicting_supplemental"):
        rebuild.supplemental_arrays(panel, roots)
    with pytest.raises(ValueError, match="missing_source_api"):
        list(rebuild.source_records(roots, "unavailable"))
    assert len(list(rebuild.source_records(roots, "daily_basic"))) == 3


def test_rebuilt_signals_equal_public_function_and_units(tmp_path):
    """Source hands/thousands/10k amounts become the exact public signal function inputs."""
    from stockops.screen.return80.signals import stock_signals

    panel, source, roots, benchmark = fixture(tmp_path)
    matrices, st = rebuild.build_signals(panel, source, roots, benchmark, tmp_path / "signals")
    values = panel.values
    dates = pd.to_datetime(panel.calendar)
    frame = pd.DataFrame(
        {
            name: values["raw_" + name][:, 0]
            for name in ("open", "high", "low", "close", "adj_factor", "up_limit")
        },
        index=dates,
    )
    frame["volume"] = values["raw_vol"][:, 0] * 100
    frame["amount"] = values["raw_amount"][:, 0] * 1000
    frame["pre_close"] = frame["volume_ratio"] = 1.0
    frame["main_amount"] = 0.0
    benchmark_frame = pd.DataFrame(rebuild.read_envelope(benchmark))
    benchmark_frame["date"] = dates
    benchmark_frame["volume"] = benchmark_frame.vol
    expected = stock_signals(frame, benchmark_frame, dates)
    for name, actual in matrices.items():
        np.testing.assert_array_equal(actual[:, 0], expected[name].to_numpy())
    assert (st == 0).all()
    axis = {"days": np.array([149]), "securities": np.array([0]), "common": np.array([True])}
    patterns = np.zeros((150, 1), dtype=np.uint32)
    patterns[149, 0] = 1
    values, names = rebuild.baseline_features(
        panel, source, matrices, axis, patterns, np.full((150, 1), 12.34, dtype=np.float32)
    )
    assert values.shape == (1, 36) and values.dtype == np.float32
    assert names[16] == "signal:volatility_contraction_breakout"
    assert values[0, 16] == 1 and values[0, 17:].sum() == 0


def test_candidate_scope_common_and_rounded_old_scores():
    """Original scope keeps ChiNext, excludes STAR; common failure stays on candidate axis."""
    from stockops.screen.return80.signals import NAMES

    codes = ["000001.SZ", "300001.SZ", "688001.SH"]
    shape = (1, 3)
    matrices = {name: np.zeros(shape, dtype=bool) for name in NAMES}
    matrices.update({name: np.ones(shape) for name in rebuild.SIGNAL_FIELDS})
    matrices["oversold_rebound"][:] = True
    matrices["capital_oversold_accumulation"][:] = True
    matrices["average_amount"][:] = matrices["universe_amount"][:] = 100000000
    matrices["reference"][:] = 10
    matrices["stop"][:] = 8
    matrices["technical_risk"][:] = 0
    panel = SimpleNamespace(
        calendar=["20211018"],
        codes=codes,
        values={
            "raw_close": np.full(shape, 10.0),
            "valid": np.ones(shape),
            "beta": np.ones(shape),
            "capitalization": np.ones(shape),
            "sector_index": np.zeros(shape),
            "raw_amount": np.full(shape, 100000.0),
            "raw_up_limit": np.full(shape, 11.0),
            "raw_down_limit": np.full(shape, 9.0),
        },
    )
    panel.values["beta"][0, 1] = np.nan
    master = [
        dict(ts_code=code, list_date="20000101", delist_date="", list_status="L") for code in codes
    ]
    members = [
        dict(ts_code=code, l2_code="801010", in_date="20000101", out_date=None) for code in codes
    ]
    axis, eligible, _, scores = rebuild.candidate_axis(
        panel, matrices, np.zeros(shape), master, members
    )
    assert axis["securities"].tolist() == [0, 1]
    assert axis["common"].tolist() == [True, False]
    assert eligible.tolist() == [[True, True, False]] and scores.dtype == np.float32
    with pytest.raises(ValueError, match="conflicting_security_master"):
        rebuild.candidate_axis(
            panel,
            matrices,
            np.zeros(shape),
            master + [dict(master[0], list_date="20000102")],
            members,
        )


def test_original_label_statuses_and_no_zero_fill():
    """Labels retain common failures, terminal censoring and actual independent-trade outcomes."""
    panel, _ = allocation_fixture()
    axis = {
        "days": np.array([0, 1, 7]),
        "securities": np.zeros(3, dtype=int),
        "common": np.array([True, False, True]),
    }
    labels = rebuild.build_labels(panel, axis, [])
    assert labels[1, 3] == 0 and np.isnan(labels[1, :3]).all()
    assert labels[2, 3] == 2 and np.isnan(labels[2, :3]).all()
    assert labels[0, 3] in (1, 2, 3)


def test_complete_rebuild_has_47_columns_and_relocatable_replay(tmp_path):
    """Fresh synthetic raw history runs all reconstruction phases without private scripts."""
    from datetime import date, timedelta

    from tushare_entitlement_hub.return80_research.dataset import load_dataset
    from tushare_entitlement_hub.strategy_discovery.replay_inputs import verified_replay_manifest

    panel, source, roots, benchmark = fixture(tmp_path, 300)
    old_calendar = panel.calendar
    panel.calendar = [
        (date(2021, 10, 18) + timedelta(days=index)).strftime("%Y%m%d") for index in range(300)
    ]
    dates = dict(zip(old_calendar, panel.calendar, strict=True))
    for path in [benchmark, *roots[0].rglob("*.json")]:
        text = path.read_text()
        for before, after in dates.items():
            text = text.replace(before, after)
        path.write_text(text)
    panel_root = tmp_path / "panel"
    panel_root.mkdir()
    panel.market = {
        key: np.arange(100, 400, dtype=float)
        for key in ("index", "total_index", "index_open", "amount")
    }
    panel.market.update({key: np.full(300, 0.5) for key in ("breadth", "advancing", "limit_down")})
    for group in ("values", "signals", "market"):
        for name, value in getattr(panel, group).items():
            np.save(panel_root / (group + "-" + name + ".npy"), value)
    (panel_root / "panel.json").write_text(
        json.dumps(
            {
                "calendar": panel.calendar,
                "codes": panel.codes,
                **{group: list(getattr(panel, group)) for group in ("values", "signals", "market")},
            }
        )
    )
    (source / "dataset.json").write_text(
        json.dumps(
            {"calendar": panel.calendar, "securities": panel.codes, "source_fingerprint": "f" * 64}
        )
    )
    envelope(
        roots[0] / "stock_basic" / "master.json",
        "stock_basic",
        [dict(ts_code="000001.SZ", list_status="L", list_date="19900101", delist_date="")],
    )
    members = tmp_path / "members.json"
    envelope(
        members,
        "index_member_all",
        [dict(ts_code="000001.SZ", l2_code="801011.SI", in_date="19900101", out_date=None)],
    )
    for api, fields in {
        "cyq_perf": [
            "ts_code",
            "trade_date",
            "cost_5pct",
            "cost_50pct",
            "cost_95pct",
            "weight_avg",
        ],
        "forecast_vip": ["ts_code", "ann_date", "end_date", "p_change_min", "p_change_max"],
        "fina_indicator_vip": [
            "ts_code",
            "ann_date",
            "end_date",
            "dt_netprofit_yoy",
            "netprofit_yoy",
        ],
        "income_vip": [
            "ts_code",
            "ann_date",
            "f_ann_date",
            "end_date",
            "report_type",
            "comp_type",
            "n_income_attr_p",
        ],
    }.items():
        envelope(roots[0] / api / "empty.json", api, [], fields=fields)
    distributions = tmp_path / "distributions"
    distributions.mkdir()
    envelope(distributions / "dividend-fixture.json", "dividend", [], fields=["ts_code"])
    spec = tmp_path / "rebuild.json"
    spec.write_text(
        json.dumps(
            {
                "schema": "return80-rebuild/v1",
                "panel": "panel",
                "source_dataset": "source",
                "raw_roots": ["raw"],
                "benchmark": "benchmark.json",
                "members": ["members.json"],
                "distributions": "distributions",
            }
        )
    )
    result = rebuild.rebuild_matrices(spec, tmp_path / "rebuilt")
    arrays, manifest = load_dataset(result.parent)
    assert arrays["X"].shape[1] == 47 and len(arrays["fit_indices"]) > 0
    assert manifest["historical_pit_certified"] is False
    assert np.isnan(arrays["X"][:, 36:]).all()
    verified_replay_manifest(result.parent.parent / "replay")
    with pytest.raises(ValueError, match="existing_output"):
        rebuild.rebuild_matrices(spec, tmp_path / "rebuilt")
