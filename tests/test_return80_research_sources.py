"""Public plan, raw preservation and historical panel adapter integration."""

import json

import pytest

from tushare_entitlement_hub.client import Client, FakeTransport
from tushare_entitlement_hub.raw import RawStore
from tushare_entitlement_hub.return80_research.collection import collect_plan, load_collection_plan
from tushare_entitlement_hub.return80_research.sources import (
    make_plan,
    materialize_sources,
    prepare_sources,
    request,
)
from tushare_entitlement_hub.strategy_discovery.replay_inputs import file_digest


def collect_fixture(root, requests, rows):
    """Use real paging/receipts with offline responses; small fixtures terminate on first page."""
    root.mkdir()
    plan = root / "plan.json"
    plan.write_text(json.dumps({"schema": "return80-collection-plan/v1", "requests": requests}))
    responses = [
        {
            "code": 0,
            "data": {
                "fields": item["fields"],
                "items": [[row.get(name) for name in item["fields"]] for row in records],
            },
        }
        for item, records in zip(requests, rows, strict=True)
    ]
    collect_plan(Client(FakeTransport(responses), RawStore(root / "raw")), plan, root)
    return root


def bootstrap_fixture(root):
    """Minimal dated source catalogs include one stock and one historical SW industry."""
    return collect_fixture(
        root,
        [
            request("trade_cal", {"exchange": "SSE"}),
            request("stock_basic", {"list_status": "L"}),
            request("index_classify", {"level": "L1", "src": "SW2021"}),
        ],
        [
            [
                {"exchange": "SSE", "cal_date": day, "is_open": 1, "pretrade_date": "20261016"}
                for day in ("20261019", "20261020")
            ],
            [{"ts_code": "000001.SZ", "list_date": "19900101", "list_status": "L"}],
            [{"index_code": "801010.SI", "level": "L1", "src": "SW2021"}],
        ],
    )


def test_plans_require_valid_dates_and_use_explicit_historical_partitions(tmp_path):
    """A verified catalog expands historical dates and both current/past memberships."""
    plan = make_plan(tmp_path / "bootstrap-plan.json", start="20200101", end="20261020")
    assert len(load_collection_plan(plan)["requests"]) == 5
    bootstrap = bootstrap_fixture(tmp_path / "bootstrap")
    full = make_plan(tmp_path / "full.json", start="20261019", end="20261020", bootstrap=bootstrap)
    requests = load_collection_plan(full)["requests"]
    membership = [
        item["parameters"]["is_new"] for item in requests if item["api"] == "index_member_all"
    ]
    assert membership == ["Y", "N"]
    assert any(
        item["api"] == "income_vip" and item["parameters"]["period"] == "20241231"
        for item in requests
    )
    assert len([item for item in requests if item["api"] == "daily"]) == 2
    assert all("cashflow" not in item["api"] for item in requests)
    for start, end in [("bad", "20261020"), ("20261021", "20261020")]:
        with pytest.raises(ValueError, match="plan_dates"):
            make_plan(tmp_path / "bad", start=start, end=end)
    with pytest.raises(ValueError, match="existing_output"):
        make_plan(full, start="20200101", end="20261020")


def test_source_adapter_to_history_and_panel(tmp_path):
    """Fresh offline pages become a verified panel beyond the old hard-coded 10/16 cutoff."""
    bootstrap = bootstrap_fixture(tmp_path / "bootstrap")
    requests, rows = [], []
    for day in ("20261019", "20261020"):
        for api in ("daily", "adj_factor", "daily_basic", "stk_limit"):
            item = request(api, {"trade_date": day})
            requests.append(item)
            rows.append(
                [
                    {
                        **dict.fromkeys(item["fields"], 1.0),
                        "ts_code": "000001.SZ",
                        "trade_date": day,
                        "close": 10.0,
                        "open": 10.0,
                        "high": 10.0,
                        "low": 10.0,
                        "up_limit": 11.0,
                        "down_limit": 9.0,
                        "amount": 100000.0,
                        "total_mv": 10000.0,
                    }
                ]
            )
    for api, parameters, records in [
        (
            "index_member_all",
            {"l1_code": "801010.SI", "is_new": "Y"},
            [
                {
                    "ts_code": "000001.SZ",
                    "l1_code": "801010.SI",
                    "l2_code": "801011.SI",
                    "l3_code": "8010111.SI",
                    "in_date": "19900101",
                    "out_date": None,
                    "is_new": "Y",
                }
            ],
        ),
        (
            "sw_daily",
            {"ts_code": "801010.SI"},
            [
                {"ts_code": "801010.SI", "trade_date": day, "close": 100.0, "amount": 1000.0}
                for day in ("20261019", "20261020")
            ],
        ),
        ("dividend", {"ts_code": "000001.SZ"}, []),
    ]:
        requests.append(request(api, parameters))
        rows.append(records)
    for code in ("000985.CSI", "H00985.CSI", "000300.SH"):
        requests.append(request("index_daily", {"ts_code": code}))
        rows.append(
            [
                {"ts_code": code, "trade_date": day, "open": 100.0, "close": 100.0, "vol": 100.0}
                for day in ("20261019", "20261020")
            ]
        )
    collected = collect_fixture(tmp_path / "collection", requests, rows)
    sources = tmp_path / "sources"
    materialize_sources(bootstrap, collected, sources)
    source_manifest = json.loads((sources / "sources.json").read_text())
    assert "market/index-000300.SH.json" in source_manifest["files"]
    assert (sources / "market/dividend-000001.SZ.json").exists()
    result = prepare_sources(sources, tmp_path / "prepared")
    assert result.exists()
    panel = json.loads((result.parent / "panel/panel.json").read_text())
    assert panel["calendar"] == ["20261019", "20261020"]
    assert panel["codes"] == ["000001.SZ"]
    with pytest.raises(ValueError, match="fresh_source"):
        materialize_sources(bootstrap, collected, sources)
    with pytest.raises(ValueError, match="fresh_prepared"):
        prepare_sources(sources, result.parent)
    (sources / "market/index-000300.SH.json").write_text("{}")
    with pytest.raises(ValueError, match="changed_materialized"):
        prepare_sources(sources, tmp_path / "tampered")


def test_changed_raw_page_rejected_before_normalization(tmp_path):
    """A receipt hash mismatch cannot silently become a new historical snapshot."""
    bootstrap = bootstrap_fixture(tmp_path / "bootstrap")
    collection = collect_fixture(
        tmp_path / "collection", [request("daily", {"trade_date": "20261019"})], [[]]
    )
    manifest = json.loads((collection / "collection.json").read_text())
    raw = collection / manifest["partitions"][0]["files"][0]["path"]
    raw.write_text("{}")
    assert file_digest(raw) != manifest["partitions"][0]["files"][0]["sha256"]
    with pytest.raises(ValueError, match="changed_source_raw"):
        materialize_sources(bootstrap, collection, tmp_path / "output")
