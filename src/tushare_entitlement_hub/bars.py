"""通过已有 HTTPS 原始层组合日周月行情，不依赖 SDK 的明文 HTTP 默认路径。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .pit import adjust_prices

if TYPE_CHECKING:
    from .client import Client, Page

BAR_APIS = {
    ("E", "D"): "daily",
    ("E", "W"): "weekly",
    ("E", "M"): "monthly",
    ("I", "D"): "index_daily",
    ("I", "W"): "index_weekly",
    ("I", "M"): "index_monthly",
    ("FT", "D"): "fut_daily",
    ("O", "D"): "opt_daily",
    ("FD", "D"): "fund_daily",
}


def compose_bars(client: Client, parameters: dict[str, Any], observed_at: str) -> Page:
    """组合指定证券的行情、固定锚点复权、换手量比和区间内均线，并保留底层来源。

    输入 Client、ts_code/start_date/end_date 及可选 asset/freq/adj/anchor_factor/ma/factors。
    输出 Page，底层原始响应逐页先落盘，派生响应保存 source_hashes。
    示例：asset=E、freq=D 调 daily；adj=qfq、anchor_factor=4 时价格10、因子2得到5。
    均线只使用请求区间已有历史，窗口不足为 None；不伪造区间外预热数据。
    """
    from .catalog import load_policies
    from .client import ProviderError

    allowed = {
        "ts_code",
        "start_date",
        "end_date",
        "asset",
        "freq",
        "adj",
        "anchor_factor",
        "ma",
        "factors",
        "exchange",
    }
    asset = str(parameters.get("asset", "E")).upper()
    frequency = str(parameters.get("freq", "D")).upper()
    api_name = BAR_APIS.get((asset, frequency))
    if api_name is None or set(parameters) - allowed:
        raise ProviderError("contract_failure", "组合行情资产、频率或参数不受支持")
    if not all(parameters.get(field) for field in ("ts_code", "start_date", "end_date")):
        raise ProviderError("contract_failure", "组合行情需要证券及明确日期范围")
    windows = parameters.get("ma") or []
    factors = parameters.get("factors") or []
    if (
        not isinstance(windows, list)
        or any(type(value) is not int or value < 1 for value in windows)
        or not isinstance(factors, list)
        or any(value not in {"tor", "vr"} for value in factors)
    ):
        raise ProviderError("contract_failure", "组合行情均线或指标参数无效")
    adjustment = parameters.get("adj")
    if adjustment is not None and (
        asset not in {"E", "FD"}
        or adjustment not in {"qfq", "hfq"}
        or "anchor_factor" not in parameters
    ):
        raise ProviderError("contract_failure", "复权需要支持的资产和显式固定锚点")
    if factors and (asset, frequency) != ("E", "D"):
        raise ProviderError("contract_failure", "换手率和量比仅支持股票日线")
    request = {field: parameters[field] for field in ("ts_code", "start_date", "end_date")}
    if parameters.get("exchange"):
        request["exchange"] = parameters["exchange"]
    policies = load_policies()
    hashes: list[str] = []

    def fetch_rows(name: str) -> list[dict[str, Any]]:
        """按底层接口真实分页政策取完整分片；例如 daily 返回行列表并登记原始哈希。"""
        policy = policies[name]
        pages = client.pages(
            name,
            request,
            observed_at,
            policy.primary_key,
            policy.page_size,
            policy.max_pages,
            policy.supports_pagination,
        )
        hashes.extend(page.raw.response_hash for page in pages)
        return [dict(row) for page in pages for row in page.rows]

    rows = sorted(fetch_rows(api_name), key=lambda row: str(row["trade_date"]))
    if adjustment is not None and rows:
        factor_api = "fund_adj" if asset == "FD" else "adj_factor"
        by_date = {str(row["trade_date"]): row["adj_factor"] for row in fetch_rows(factor_api)}
        try:
            daily_factors = [float(by_date[str(row["trade_date"])]) for row in rows]
            for field in ("open", "high", "low", "close", "pre_close"):
                adjusted = adjust_prices(
                    [float(row[field]) for row in rows],
                    daily_factors,
                    float(parameters["anchor_factor"]),
                    adjustment,
                )
                for row, value in zip(rows, adjusted, strict=True):
                    row[field] = value
        except (KeyError, TypeError, ValueError):
            raise ProviderError("contract_failure", "复权价格、因子或固定锚点不完整") from None
        for row in rows:
            row["change"] = row["close"] - row["pre_close"]
            row["pct_chg"] = 100 * row["change"] / row["pre_close"] if row["pre_close"] else None
    if factors:
        indicators = {str(row["trade_date"]): row for row in fetch_rows("daily_basic")}
        for row in rows:
            for factor, field in (("tor", "turnover_rate"), ("vr", "volume_ratio")):
                if factor in factors:
                    row[field] = indicators.get(str(row["trade_date"]), {}).get(field)
    try:
        for index, row in enumerate(rows):
            for window in windows:
                history = rows[index - window + 1 : index + 1] if index + 1 >= window else []
                for field, prefix in (("close", "ma"), ("vol", "ma_v_")):
                    row[f"{prefix}{window}"] = (
                        sum(float(item[field]) for item in history) / window if history else None
                    )
    except (KeyError, ValueError, TypeError):
        raise ProviderError("contract_failure", "均线输入不是有效数值") from None
    fields = list(rows[0]) if rows else ["ts_code", "trade_date", "open", "high", "low", "close"]
    response = {
        "code": 0,
        "data": {
            "fields": fields,
            "items": [[row[key] for key in fields] for row in reversed(rows)],
        },
        "derivation": {
            "method": "pro_bar_https_v1",
            "source_hashes": hashes,
            "anchor_factor": parameters.get("anchor_factor"),
            "warmup": "requested_range_only",
        },
    }
    return client._parse(client.raw_store.write("pro_bar", parameters, response, observed_at))
