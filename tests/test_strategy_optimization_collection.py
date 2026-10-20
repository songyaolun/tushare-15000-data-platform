"""公开历史补采的缓存、失败、限额和显式联网边界。"""

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from tushare_entitlement_hub.client import ProviderError
from tushare_entitlement_hub.strategy_optimization import cli, collection


def response(rows):
    """输入公开字典列表，输出无凭据供应商成功包络；空列表产生空items。"""
    fields = list(rows[0]) if rows else ["unused"]
    return {
        "code": 0,
        "data": {"fields": fields, "items": [[row[name] for name in fields] for row in rows]},
    }


def test_cached_response_preserves_raw_and_retries(tmp_path, monkeypatch):
    """场景：先失败再成功及持久失败；预期：成功原文留存、缓存复用、三次后抛原安全错误。"""
    calls = []
    monkeypatch.setattr(collection.time, "sleep", lambda delay: None)

    def request(api, parameters):
        """记录调用并模拟首次瞬态故障；后续输入返回单日公开记录。"""
        calls.append(api)
        if len(calls) == 1:
            raise ProviderError("transient_failure", "transient")
        return response([{"trade_date": "20241220"}])

    transport = SimpleNamespace(request=request)
    arguments = (transport, tmp_path, "cached.json", "api", {"trade_date": "20241220"}, 5)
    assert collection.cached_response(*arguments) == [{"trade_date": "20241220"}]
    assert collection.cached_response(*arguments) == [{"trade_date": "20241220"}]
    assert len(calls) == 2
    with pytest.raises(ValueError, match="request_mismatch"):
        collection.cached_response(transport, tmp_path, "cached.json", "wrong", {}, 5)
    with pytest.raises(ValueError, match="truncation"):
        collection.cached_response(*arguments[:-1], 1)

    def failure(*args):
        """输入任意请求，始终返回安全ProviderError以检查重试上限。"""
        raise ProviderError("transient_failure", "offline")

    with pytest.raises(ProviderError):
        collection.cached_response(
            SimpleNamespace(request=failure), tmp_path, "failure.json", "api", {}, 5
        )
    with pytest.raises(ValueError, match="date_mismatch"):
        collection.cached_response(
            transport, tmp_path, "bad-date.json", "api", {"trade_date": "20241221"}, 5
        )


def test_history_collectors_preserve_partitions_and_same_day_ranking(tmp_path, monkeypatch):
    """场景：一个行业和一日概念；预期：Y/N历史分区、涨幅换手同日排序、完成清单正确。"""
    calls = []
    monkeypatch.setattr(collection.time, "sleep", lambda delay: None)

    def request(api, parameters):
        """输入不同历史接口，输出确定性公开数据，记录实际分区条件供断言。"""
        calls.append((api, parameters))
        if api == "index_classify":
            return response([{"index_code": "X"}])
        if api == "dc_index":
            return response(
                [
                    {"trade_date": "20241220", "ts_code": "A", "pct_change": 2, "turnover_rate": 3},
                    {"trade_date": "20241220", "ts_code": "B", "pct_change": 1, "turnover_rate": 2},
                ]
            )
        return response([])

    transport = SimpleNamespace(request=request)
    collection.collect_industries(transport, tmp_path / "industry")
    path = collection.collect_concepts(transport, tmp_path / "concepts", ["20240101", "20241220"])
    assert json.loads(path.read_text())["days"][0]["chosen"] == ["A", "B"]
    assert [params["is_new"] for api, params in calls if api == "index_member_all"] == ["Y", "N"]
    assert (
        collection.active_concepts(
            [{"pct_change": None}, {"pct_change": float("nan"), "turnover_rate": 1}]
        )
        == []
    )


@pytest.mark.parametrize("wrapped_calendar", [False, True])
def test_collection_cli_requires_live_and_keeps_credentials_in_transport(
    tmp_path,
    monkeypatch,
    wrapped_calendar,
):
    """场景：显式联网开关及两种日历包络；预期：无开关拒绝、有开关仅传给既有HTTPS传输。"""
    calendar = tmp_path / "calendar.json"
    calendar.write_text(json.dumps({"trading_dates": []} if wrapped_calendar else []))
    observed = []

    class Transport:
        """模拟上下文传输并检查凭据只由提供函数读取，不写入任何产物。"""

        def __init__(self, credential, live):
            """输入凭据回调与live；输出无网络的测试实例并记录显式开关。"""
            observed.append((credential(), live))

        def __enter__(self):
            """无参数，返回当前测试传输。"""
            return self

        def __exit__(self, *args):
            """输入退出上下文参数，输出None且不吞异常。"""
            return None

    monkeypatch.setattr(cli, "HttpTransport", Transport)
    monkeypatch.setattr(cli, "configured_credential", lambda path: "synthetic")
    monkeypatch.setattr(cli, "collect_industries", lambda *args: None)
    monkeypatch.setattr(cli, "collect_concepts", lambda *args: None)
    arguments = ["collect-history", "secret", str(calendar), str(tmp_path / "output")]
    assert CliRunner().invoke(cli.optimization_app, arguments).exit_code != 0
    result = CliRunner().invoke(cli.optimization_app, arguments + ["--live"])
    assert result.exit_code == 0, result.output
    assert observed == [("synthetic", True)]
    assert "synthetic" not in result.output
