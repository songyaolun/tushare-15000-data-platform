"""Offline collection and new daily features use the real paging and raw storage contracts."""

import json

import numpy as np
import pytest
from test_return80_research_dataset import arrays_fixture

from tushare_entitlement_hub.client import Client, FakeTransport, ProviderError
from tushare_entitlement_hub.raw import RawStore
from tushare_entitlement_hub.return80_research.cli import main
from tushare_entitlement_hub.return80_research.collection import (
    collect_plan,
    load_collection_plan,
    read_collection,
)
from tushare_entitlement_hub.return80_research.dataset import (
    load_dataset,
    save_dataset,
)
from tushare_entitlement_hub.return80_research.features import (
    build_daily_block,
    daily_feature_values,
)


def plan_fixture(root):
    """Write a single explicit daily partition, with page size two and no network credentials."""
    path = root / "plan.json"
    path.write_text(
        json.dumps(
            {
                "schema": "return80-collection-plan/v1",
                "requests": [
                    {
                        "api": "daily_basic",
                        "parameters": {"trade_date": "20211018"},
                        "fields": ["ts_code", "trade_date", "turnover_rate"],
                        "primary_key": ["ts_code", "trade_date"],
                        "page_size": 2,
                        "maximum_pages": 3,
                    }
                ],
            }
        )
    )
    return path


def response(rows):
    """Encode fixture rows in the actual Tushare field/item envelope; [] means a terminal page."""
    return {
        "code": 0,
        "data": {"fields": ["ts_code", "trade_date", "turnover_rate"], "items": rows},
    }


def test_collection_resume_build_append_end_to_end(tmp_path, capsys):
    """Download fixture pages, relocate their collection, derive a factor and append by identity."""
    plan = plan_fixture(tmp_path)
    root = tmp_path / "collection"
    transport = FakeTransport(
        [response([["A", "20211018", 2], ["B", "20211018", 4]]), response([])]
    )
    client = Client(transport, RawStore(root / "raw"))
    collect_plan(client, plan, root)
    collect_plan(Client(FakeTransport([]), RawStore(root / "raw")), plan, root)
    assert len(read_collection(root, "daily_basic")) == 2
    source = tmp_path / "data"
    save_dataset(source, arrays_fixture(), {})
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "return80-daily-features/v1",
                "features": [
                    {
                        "name": "turnover_fraction",
                        "api": "daily_basic",
                        "field": "turnover_rate",
                        "scale": 0.01,
                        "unit": "fraction",
                        "formula_version": "turnover/v1",
                    }
                ],
            }
        )
    )
    main(
        [
            "build-daily-features",
            "--dataset",
            str(source),
            "--collection",
            str(root),
            "--recipe",
            str(recipe),
            "--output",
            str(tmp_path / "block"),
        ]
    )
    main(
        [
            "append-features",
            "--dataset",
            str(source),
            "--block",
            str(tmp_path / "block/block.json"),
            "--output",
            str(tmp_path / "extended"),
        ]
    )
    arrays, _ = load_dataset(tmp_path / "extended")
    np.testing.assert_allclose(arrays["X"][0], [1.0, 0.02])
    assert np.isnan(arrays["X"][1:, 1]).all()
    assert "completed" in capsys.readouterr().out
    with pytest.raises(ValueError, match="missing_collection_api"):
        read_collection(root, "daily")
    with pytest.raises(ValueError, match="fresh_feature_output"):
        build_daily_block(source, root, recipe, tmp_path / "block")
    with pytest.raises(ValueError, match="live_collection_requires_explicit_flag"):
        main(["collect", "--plan", str(plan), "--output", str(tmp_path / "live")])
    manifest = json.loads((root / "collection.json").read_text())
    raw = root / manifest["partitions"][0]["files"][0]["path"]
    raw.write_text("{}")
    with pytest.raises(ValueError, match="collection_raw_changed"):
        read_collection(root, "daily_basic")
    with pytest.raises(ValueError, match="collection_raw_changed"):
        collect_plan(client, plan, root)


@pytest.mark.parametrize(
    "rows",
    [
        [["A", "20211019", 1]],
        [["A", "20211018", 1], ["A", "20211018", 2]],
    ],
)
def test_collection_date_mismatch_or_duplicate_fails(tmp_path, rows):
    """Wrong date and repeated identity fail before a complete collection marker is written."""
    root = tmp_path / "raw"
    client = Client(FakeTransport([response(rows), response([])]), RawStore(root / "raw"))
    with pytest.raises((ValueError, ProviderError)):
        collect_plan(client, plan_fixture(tmp_path), root)
    assert not (root / "collection.json").exists()


def test_collection_plan_tampering_and_missing_fields(tmp_path):
    """Invalid plans fail before a request; a changed plan cannot reuse previous receipts."""
    path = plan_fixture(tmp_path)
    plan = json.loads(path.read_text())
    plan["requests"].append(plan["requests"][0])
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="invalid_collection_request"):
        load_collection_plan(path)
    path.write_text("{}")
    with pytest.raises(ValueError, match="invalid_collection_plan"):
        load_collection_plan(path)


@pytest.mark.parametrize(
    "operation,window,expected",
    [
        ("value", 1, [1, 2, 4]),
        ("mean", 2, [np.nan, 1.5, 3]),
        ("std", 2, [np.nan, 0.5, 1]),
        ("change", 1, [np.nan, 1, 2]),
        ("return", 1, [np.nan, 1, 1]),
    ],
)
def test_daily_feature_formulas_and_lag(operation, window, expected):
    """All daily operators use only the current/past market sessions, with explicit lag."""
    calendar = ["20211018", "20211019", "20211020"]
    rows = [
        {"ts_code": "A", "trade_date": day, "x": value}
        for day, value in zip(calendar, [1, 2, 4], strict=True)
    ]
    values = daily_feature_values(
        rows, calendar, ["A"], field="x", window=window, operation=operation
    )
    np.testing.assert_allclose(values[:, 0], expected, equal_nan=True)
    lagged = daily_feature_values(rows, calendar, ["A"], field="x", lag=1)
    np.testing.assert_allclose(lagged[:, 0], [np.nan, 1, 2], equal_nan=True)
    rows[1]["available"] = "20211020"
    rows[0]["available"] = "20211018"
    rows[2]["available"] = "20211020"
    known = daily_feature_values(
        rows, calendar, ["A"], field="x", available_field="available", operation="mean", window=2
    )
    assert np.isnan(known).all()


@pytest.mark.parametrize("damage", ["duplicate", "missing", "inf", "bad_date", "overflow"])
def test_daily_feature_invalid_inputs(damage):
    """No duplicate, missing field, nonfinite number, invalid date or float32 overflow is hidden."""
    rows = [{"ts_code": "A", "trade_date": "20211018", "x": 1}]
    if damage == "duplicate":
        rows += rows
    elif damage == "missing":
        del rows[0]["x"]
    elif damage == "inf":
        rows[0]["x"] = np.inf
    elif damage == "bad_date":
        rows[0]["trade_date"] = "20210230"
    else:
        rows[0]["x"] = 1e100
    with pytest.raises(ValueError):
        daily_feature_values(rows, ["20211018"], ["A"], field="x")
    with pytest.raises(ValueError, match="invalid_daily_feature_recipe"):
        daily_feature_values([], ["20211018"], ["A"], field="x", window=0)
