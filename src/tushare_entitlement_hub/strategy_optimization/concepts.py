"""按每日历史成员重建活跃概念群动，作为独立特征附加层。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dataset import read_envelope


def load_features(dataset: Path, code: str) -> dict[str, np.ndarray]:
    """加载股票基础特征及可选概念附加层；例如无附加文件时只返回原有技术指标。"""
    with np.load(dataset / f"{code}.npz", allow_pickle=False) as saved:
        features = {name: saved[name] for name in saved.files}
    overlay = dataset / "concepts" / f"{code}.npz"
    if overlay.exists():
        with np.load(overlay, allow_pickle=False) as saved:
            features.update({name: saved[name] for name in saved.files})
    return features


def attach_concepts(dataset: Path, concept_root: Path, output: Path) -> Path:
    """给完整基础数据集附加每日概念热度和剔除本股群动，基础文件以只读链接复用。

    输入每日前五活跃概念采集清单及成员；输出新数据集。至少五只同伴、90%报价
    覆盖、60%上涨且平均收益为正才通过群动；无历史日期保持False，不向过去填充。
    例：六只成员全部上涨，六只各自concept_group=True；单只概念不能通过。
    """
    manifest = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    collection = json.loads((concept_root / "collection.json").read_text(encoding="utf-8"))
    if output.exists():
        raise ValueError("concept_output_exists")
    calendar, codes = manifest["calendar"], manifest["securities"]
    calendar_index = {day: index for index, day in enumerate(calendar)}
    code_index = {code: index for index, code in enumerate(codes)}
    returns = np.full((len(codes), len(calendar)), np.nan)
    for index, code in enumerate(codes):
        with np.load(dataset / f"{code}.npz", allow_pickle=False) as saved:
            returns[index] = saved["return1"]
    hot = np.zeros(returns.shape, dtype=bool)
    group = np.zeros(returns.shape, dtype=bool)
    observed_days = []
    for record in collection["days"]:
        day = record["trade_date"]
        if day not in calendar_index or day in observed_days:
            raise ValueError("concept_calendar_invalid")
        observed_days.append(day)
        position = calendar_index[day]
        for concept in record["chosen"]:
            rows = read_envelope(concept_root / f"members-{day}-{concept}.json")
            if any(row["trade_date"] != day or row["ts_code"] != concept for row in rows):
                raise ValueError("concept_membership_partition_mismatch")
            members = sorted(
                {
                    str(row["con_code"])
                    for row in rows
                    if str(row["con_code"]).endswith((".SH", ".SZ"))
                }
            )
            indices = np.array(
                [code_index[code] for code in members if code in code_index], dtype=int
            )
            current = returns[indices, position]
            valid = np.isfinite(current)
            peers = int(valid.sum()) - valid.astype(int)
            expected = len(members) - 1
            hot[indices, position] = True
            if expected < 5:
                continue
            breadth = (np.sum(current > 0) - (current > 0)) / np.maximum(peers, 1)
            mean_return = (np.nansum(current) - np.nan_to_num(current)) / np.maximum(peers, 1)
            group[indices, position] |= (
                valid
                & (peers >= 5)
                & (peers / expected >= 0.9)
                & (breadth >= 0.6)
                & (mean_return > 0)
            )
    expected_days = [day for day in calendar if "20241220" <= day <= "20261016"]
    if sorted(observed_days) != expected_days:
        raise ValueError("concept_collection_incomplete")
    output.mkdir(parents=True)
    (output / "concepts").mkdir()
    for index, code in enumerate(codes):
        (output / f"{code}.npz").symlink_to((dataset / f"{code}.npz").resolve())
        np.savez_compressed(
            output / "concepts" / f"{code}.npz", concept_hot=hot[index], concept_group=group[index]
        )
    manifest["concept_root"] = str(concept_root)
    manifest["concept_history_start"] = "20241220"
    manifest["concept_history_days"] = len(observed_days)
    manifest["concept_selection"] = collection["selection"]
    destination = output / "dataset.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination
