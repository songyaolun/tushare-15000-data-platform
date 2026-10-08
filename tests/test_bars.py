"""组合行情仅使用离线原始响应，验证价格、均线、来源和参数边界。"""

import pytest

from tushare_entitlement_hub.client import Client, FakeTransport, ProviderError
from tushare_entitlement_hub.raw import RawStore

PARAMETERS = {"ts_code": "A.SH", "start_date": "20240101", "end_date": "20240103"}


def response(fields, rows):
    """构造提供方响应；例如 ['close']、[[2]] 返回单字段数据。"""
    return {"code": 0, "data": {"fields": fields, "items": rows}}


def bars_response():
    """提供两天未复权行情；第二日close为12，pre_close为10，用于核对复权和均线。"""
    return response(
        ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol"],
        [["A.SH", "20240101", 10, 12, 9, 10, 8, 20], ["A.SH", "20240102", 11, 13, 10, 12, 10, 30]],
    )


def test_bar_composition_adjustment_factors_and_lineage(tmp_path):
    """场景：双日行情加因子和指标；预期：固定锚点价格、均线及底层来源均准确。"""
    client = Client(
        FakeTransport(
            [
                bars_response(),
                response(
                    ["ts_code", "trade_date", "adj_factor"],
                    [["A.SH", "20240101", 2], ["A.SH", "20240102", 2]],
                ),
                response(
                    ["ts_code", "trade_date", "turnover_rate", "volume_ratio"],
                    [["A.SH", "20240101", 1, 2], ["A.SH", "20240102", 3, 4]],
                ),
            ]
        ),
        RawStore(tmp_path),
    )
    page = client.fetch(
        "pro_bar",
        {
            **PARAMETERS,
            "adj": "qfq",
            "anchor_factor": 4,
            "factors": ["tor", "vr"],
            "ma": [2],
            "exchange": "SSE",
        },
        "2024-01-03T00:00:00Z",
    )
    assert page.rows[0]["close"] == 6
    assert page.rows[0]["ma2"] == 5.5 and page.rows[0]["ma_v_2"] == 25
    assert page.rows[0]["turnover_rate"] == 3 and page.rows[0]["volume_ratio"] == 4
    assert page.rows[1]["ma2"] is None
    assert len(page.raw.response["derivation"]["source_hashes"]) == 3
    assert len(list(tmp_path.rglob("*.json"))) == 4


@pytest.mark.parametrize(
    "extra",
    [
        {"asset": "X"},
        {"freq": "1MIN"},
        {"unexpected": 1},
        {"ts_code": ""},
        {"ma": [-1]},
        {"ma": "5"},
        {"factors": ["wrong"]},
        {"adj": "qfq"},
        {"asset": "I", "adj": "hfq", "anchor_factor": 1},
        {"asset": "I", "factors": ["tor"]},
    ],
)
def test_bar_parameter_errors_do_not_request(tmp_path, extra):
    """场景：不支持的资产、频率、指标或缺固定锚点；预期：发请求前明确拒绝。"""
    client = Client(FakeTransport([]), RawStore(tmp_path))
    with pytest.raises(ProviderError):
        client.fetch("pro_bar", {**PARAMETERS, **extra}, "2024-01-03T00:00:00Z")


def test_bar_empty_and_unadjusted_paths(tmp_path):
    """场景：空响应及普通指数行情；预期：空结果不伪造，未复权价格保持原值。"""
    client = Client(
        FakeTransport([response(["ts_code", "trade_date"], []), bars_response()]),
        RawStore(tmp_path),
    )
    assert not client.fetch("pro_bar", PARAMETERS, "2024-01-03T00:00:00Z").rows
    assert (
        client.fetch("pro_bar", {**PARAMETERS, "asset": "I"}, "2024-01-03T00:00:00Z").rows[0][
            "close"
        ]
        == 12
    )


def test_bar_missing_adjustment_and_invalid_mean(tmp_path):
    """场景：复权因子缺一天或均线输入无效；预期：保留底层原始数据并拒绝派生。"""
    client = Client(
        FakeTransport([bars_response(), response(["ts_code", "trade_date", "adj_factor"], [])]),
        RawStore(tmp_path),
    )
    with pytest.raises(ProviderError, match="复权"):
        client.fetch(
            "pro_bar", {**PARAMETERS, "adj": "qfq", "anchor_factor": 1}, "2024-01-03T00:00:00Z"
        )
    invalid = bars_response()
    invalid["data"]["items"][0][-1] = "invalid"
    client = Client(FakeTransport([invalid]), RawStore(tmp_path))
    with pytest.raises(ProviderError, match="均线"):
        client.fetch("pro_bar", {**PARAMETERS, "ma": [1]}, "2024-01-03T00:00:00Z")
