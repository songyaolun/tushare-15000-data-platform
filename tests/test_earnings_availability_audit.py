"""严格归档审计不能被误当作供应商历史重建的否决。"""

import gzip
import json
from datetime import datetime

import pytest

from tushare_entitlement_hub.raw import RawStore
from tushare_entitlement_hub.research_history.earnings_audit import (
    audit_snapshots,
    available_at,
    read_snapshot,
    write_audit,
)


def snapshot(root, api="forecast", rows=None):
    """建立无网络原始fixture；例snapshot(tmp_path)返回一份1月5日预告文件。"""
    rows = rows if rows is not None else [{"ts_code": "000001.SZ", "ann_date": "20260105",
        "first_ann_date": "20260105", "end_date": "20251231"}]
    fields = list(rows[0]) if rows else ["ts_code"]
    response = {"code": 0, "data": {"fields": fields,
        "items": [[row[key] for key in fields] for row in rows]}}
    return RawStore(root).write(api, {}, response, "2026-10-02T00:00:00Z").path


@pytest.mark.parametrize("api,row,observed,expected", [
    ("forecast", {"ann_date": "20260105"}, "2026-01-05T01:00:00Z", "2026-01-06T00:00:00+08:00"),
    ("forecast_vip", {"ann_date": "20260105", "first_ann_date": "20260108"},
     "2026-01-05T01:00:00Z", "2026-01-09T00:00:00+08:00"),
    ("report_rc", {"report_date": "20260105"}, "2026-01-05T01:00:00Z",
     "2026-01-06T09:15:00+08:00"),
    ("report_rc", {"report_date": "20220117", "create_time": "2022-05-02 23:21:26"},
     "2022-01-18T02:00:00Z", "2022-05-02T23:21:26+08:00"),
    ("report_rc", {"report_date": "20260105", "create_time": "2026-01-06T02:00:00Z"},
     "2026-10-02T00:00:00Z", "2026-10-02T08:00:00+08:00"),
])
def test_explicit_lags_and_observation_bounds(api, row, observed, expected):
    """批次、公告修正、供应商更新时间和观测时间取最晚，不提前到当晚19:30。"""
    assert available_at(api, row, observed).isoformat() == expected


@pytest.mark.parametrize("row,observed", [({}, "2026-01-01T00:00:00Z"),
    ({"ann_date": "bad"}, "2026-01-01T00:00:00Z"),
    ({"ann_date": "20260105"}, "2026-01-05T00:00:00"),
    ({"ann_date": "20260105", "create_time": "bad"}, "2026-01-01T00:00:00Z")])
def test_missing_public_date_or_timezone_is_unknown(row, observed):
    """缺日期、坏日期、无时区或坏更新时间必须保留未知。"""
    assert available_at("forecast", row, observed) is None


def test_unsupported_api_is_rejected(tmp_path):
    """错误接口不能混入预告审计，包括原始文件中接口不匹配。"""
    with pytest.raises(ValueError, match="unsupported_earnings_api"):
        available_at("daily", {}, "")
    with pytest.raises(ValueError, match="unsupported_earnings_api"):
        audit_snapshots([snapshot(tmp_path, "daily")], datetime.fromisoformat("2026-01-01T00:00Z"))


def test_compressed_snapshot_and_modified_hash(tmp_path):
    """压缩raw读回一致，修改响应哈希后立即拒绝。"""
    path = snapshot(tmp_path)
    compressed = tmp_path / "copy.json.gz"
    compressed.write_bytes(gzip.compress(path.read_bytes()))
    assert read_snapshot(compressed) == read_snapshot(path)
    envelope = json.loads(path.read_text())
    envelope["response_hash"] = "wrong"
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="invalid_earnings_response"):
        read_snapshot(path)


def test_bad_field_shape_rejected(tmp_path):
    """原始行字段数量不匹配不能成为看似成功的空统计。"""
    path = RawStore(tmp_path).write("forecast", {}, {"code": 0,
        "data": {"fields": ["ts_code"], "items": [[1, 2]]}}, "2026-01-01T00:00Z").path
    with pytest.raises(ValueError, match="invalid_earnings_fields"):
        read_snapshot(path)


def test_audit_dedup_cohorts_empty_api_and_no_reconstruction_rejection(tmp_path):
    """重复快照只算一次，不跨机构跨期间比较；严格归档0可得不能否决历史重建。"""
    base = {"ts_code": "000001.SZ", "report_date": "20260105", "org_name": "A",
            "quarter": "2026Q4", "create_time": "2026-01-05 21:08:00"}
    rows = [base, {**base, "report_date": "20260106"}, {**base, "org_name": "B"},
            {**base, "quarter": "2027Q4"}, {**base, "report_date": "bad"}]
    report = snapshot(tmp_path, "report_rc", rows)
    preann = snapshot(tmp_path)
    cutoff = datetime.fromisoformat("2026-08-26T19:30+08:00")
    result = audit_snapshots([report, report, preann], cutoff)
    assert result["unique_snapshots"] == 2
    assert result["apis"]["forecast"]["mature_available_records"] == 0
    assert result["apis"]["forecast_vip"]["first_available_at"] is None
    assert result["apis"]["report_rc"]["cohorts"] == 3
    assert result["apis"]["report_rc"]["cohorts_with_multiple_public_dates"] == 1
    assert result["apis"]["report_rc"]["unknown_availability_records"] == 1
    assert result["decision"] == "archive_audit_only_reconstruction_requires_separate_review"


def test_write_audit_and_timezone_requirement(tmp_path):
    """从路径清单可离线复现结果；截止时间没有时区时拒绝。"""
    path = snapshot(tmp_path)
    listing, output = tmp_path / "paths.txt", tmp_path / "audit.json"
    listing.write_text(str(path) + "\n\n")
    cutoff = datetime.fromisoformat("2026-10-03T19:30+08:00")
    result = write_audit(listing, output, cutoff)
    assert json.loads(output.read_text()) == result
    assert result["apis"]["forecast"]["mature_available_records"] == 1
    with pytest.raises(ValueError, match="mature_cutoff_requires_timezone"):
        audit_snapshots([], datetime(2026, 1, 1))
