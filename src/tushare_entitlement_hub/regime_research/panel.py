"""冻结研究面板；原始价格用于成交，复权价格只用于特征与对照分析。"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..strategy_optimization.dataset import read_envelope
from ..strategy_optimization.features import FloatArray
from ..strategy_optimization.strategies import strategy_masks
from .market import rolling_beta

FIELDS = (
    "raw_open",
    "raw_close",
    "raw_low",
    "raw_high",
    "raw_amount",
    "raw_adj_factor",
    "raw_up_limit",
    "raw_down_limit",
    "valid",
    "close",
    "ma20",
    "return1",
    "return5",
    "return20",
    "amount_ratio",
    "kdj_j",
    "sector_index",
    "sector_heat",
    "sector_breadth",
    "sector_coverage",
    "sector_peers",
    "sector_mean_return",
    "boll_position",
)


@dataclass
class Panel:
    """日×证券的冻结矩阵；例如values['raw_close'][0,0]为首日首证券原始收盘。"""

    calendar: list[str]
    codes: list[str]
    values: dict[str, FloatArray]
    signals: dict[str, FloatArray]
    market: dict[str, FloatArray]


def prepare_panel(dataset: Path, raw: Path, output: Path) -> Path:
    """把已有因果特征与新补采基准、市值组合成只读npy面板，不复制供应商原文。

    输入 FEATURES/RAW/OUT，输出OUT/panel.json；已有清单直接复用。
    月末市值仅从该日期向后延用，不做双向插值；完整Beta缺失的股票不能入选。
    """
    manifest_path = output / "panel.json"
    if manifest_path.exists():
        return manifest_path
    if output.exists() and any(output.iterdir()):
        raise ValueError("incomplete_panel_use_new_directory")
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads((dataset / "dataset.json").read_text())
    calendar, codes = source["calendar"], source["securities"]
    month_ends = {day[:6]: day for day in calendar}
    if any(not (raw / f"capitalization-{day}.json").exists() for day in month_ends.values()):
        raise ValueError("incomplete_monthly_capitalization")
    dates = {day: index for index, day in enumerate(calendar)}
    securities = {code: index for index, code in enumerate(codes)}
    shape = len(calendar), len(codes)
    values = {name: np.full(shape, np.nan) for name in (*FIELDS, "beta", "capitalization")}
    market = {}
    for code, name in (("000985.CSI", "index"), ("H00985.CSI", "total_index")):
        rows = {row["trade_date"]: row for row in read_envelope(raw / f"index-{code}.json")}
        market[name] = np.array([rows.get(day, {}).get("close", np.nan) for day in calendar])
        if name == "index":
            market["index_open"] = np.array(
                [rows.get(day, {}).get("open", np.nan) for day in calendar]
            )
    if any(not np.all(np.isfinite(array) & (array > 0)) for array in market.values()):
        raise ValueError("incomplete_benchmark_history")
    benchmark = np.r_[np.nan, market["total_index"][1:] / market["total_index"][:-1] - 1]
    signals: dict[str, FloatArray] = {}
    for column, code in enumerate(codes):
        with np.load(dataset / f"{code}.npz") as saved:
            features = {name: saved[name] for name in saved.files}
        for name in FIELDS:
            values[name][:, column] = features[name]
        values["beta"][:, column] = rolling_beta(features["return1"], benchmark)
        masks = strategy_masks(features)
        for name in (
            "volume_reversal__sector_group",
            "boll_reentry__sector_group",
            "trend_pullback__sector_group",
            "trend_breakout__sector_hot_group",
        ):
            signals.setdefault(name, np.zeros(shape, dtype=bool))[:, column] = masks[name]
    for path in sorted(raw.glob("capitalization-*.json")):
        for row in read_envelope(path):
            if row["ts_code"] in securities and row["trade_date"] in dates:
                value = row.get("total_mv")
                if isinstance(value, (float, int)) and value > 0:
                    values["capitalization"][
                        dates[row["trade_date"]], securities[row["ts_code"]]
                    ] = value
    for day in range(1, len(calendar)):
        missing = ~np.isfinite(values["capitalization"][day])
        values["capitalization"][day, missing] = values["capitalization"][day - 1, missing]
    known = np.isfinite(values["ma20"]) & (values["valid"] == 1)
    market["breadth"] = np.sum(known & (values["close"] > values["ma20"]), axis=1) / np.maximum(
        1, np.sum(known, axis=1)
    )
    valid = np.isfinite(values["return1"])
    market["advancing"] = np.sum(valid & (values["return1"] > 0), axis=1) / np.maximum(
        1, np.sum(valid, axis=1)
    )
    market["amount"] = np.nansum(values["raw_amount"], axis=1)
    limit_known = (values["valid"] == 1) & np.isfinite(values["raw_down_limit"])
    market["limit_down"] = np.sum(
        limit_known & (values["raw_close"] <= values["raw_down_limit"]), axis=1
    ) / np.maximum(1, np.sum(limit_known, axis=1))
    for category, arrays in (("values", values), ("signals", signals), ("market", market)):
        for name, array in arrays.items():
            np.save(output / f"{category}-{name}.npy", array)
    manifest_path.write_text(
        json.dumps(
            {
                "calendar": calendar,
                "codes": codes,
                "values": list(values),
                "signals": list(signals),
                "market": list(market),
                "source_fingerprint": source["source_fingerprint"],
                "historical_pit_certified": False,
            }
        )
    )
    return manifest_path


def load_panel(path: Path) -> Panel:
    """以只读内存映射加载面板；输入OUT目录，输出Panel，不请求网络。"""
    manifest = json.loads((path / "panel.json").read_text())
    groups = {
        category: {
            name: np.load(path / f"{category}-{name}.npy", mmap_mode="r")
            for name in manifest[category]
        }
        for category in ("values", "signals", "market")
    }
    return Panel(
        manifest["calendar"],
        manifest["codes"],
        groups["values"],
        groups["signals"],
        groups["market"],
    )
