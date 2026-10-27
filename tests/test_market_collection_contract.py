"""Relocated market_collection keeps public plans, requests and digests unchanged."""

import json

import tushare_entitlement_hub.market_collection.collection as market_collection
import tushare_entitlement_hub.return80_research.collection as legacy_reexport
from tushare_entitlement_hub.market_collection.fields import EXTRA_FIELDS
from tushare_entitlement_hub.return80_research.sources import request


def test_legacy_collection_module_reexports_market_collection():
    """The old import path still exposes the exact same public objects."""
    for name in (
        "collect_plan",
        "load_collection_plan",
        "read_collection",
        "verify_chips_unavailable",
    ):
        assert getattr(legacy_reexport, name) is getattr(market_collection, name)


def test_request_contract_unchanged():
    """Field declarations, paging keys and limits match the frozen collection plan."""
    assert request("cyq_perf", {"trade_date": "20260101"}) == {
        "api": "cyq_perf",
        "parameters": {"trade_date": "20260101"},
        "fields": EXTRA_FIELDS["cyq_perf"].split(","),
        "primary_key": ["ts_code", "trade_date"],
        "page_size": 2000,
        "maximum_pages": 100,
        "paginate": True,
    }
    assert request("index_member_all", {"l1_code": "801010.SI"})["primary_key"] == [
        "ts_code",
        "l1_code",
        "l2_code",
        "l3_code",
        "in_date",
    ]


def test_plan_schema_and_digest_unchanged(tmp_path):
    """Plan files keep the frozen schema string and byte-level digest behaviour."""
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "return80-collection-plan/v1",
                "requests": [request("cyq_perf", {"trade_date": "20260101"})],
            },
            indent=2,
        )
        + "\n"
    )
    loaded = market_collection.load_collection_plan(plan)
    assert loaded["schema"] == "return80-collection-plan/v1"
    copy = tmp_path / "copy.json"
    copy.write_bytes(plan.read_bytes())
    from tushare_entitlement_hub.strategy_discovery.replay_inputs import file_digest

    assert file_digest(plan) == file_digest(copy)
