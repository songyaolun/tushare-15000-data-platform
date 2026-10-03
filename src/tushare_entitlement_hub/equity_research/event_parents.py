"""P13 上游身份去重、实体索引与流式内容指纹，不保存父结果的 JSON 副本。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, fields
from typing import Any

from ..product_foundation.model import Model
from ..store import digest
from .models import ResearchInputs, ResearchResult

MAXIMUM_ENTITY_PARENTS = 64
PARENT_FAILURE_CODES = frozenset(
    {
        "research_event_parent_conflict",
        "research_event_parent_entity_missing",
        "research_event_parent_capacity_exceeded",
    }
)


@dataclass(frozen=True)
class ParentBundle:
    """同实体的全部有效父版本；只持有原对象引用与固定长度摘要，不复制载荷。"""

    parents: tuple[ResearchResult, ...]
    result_ids: tuple[str, ...]
    fingerprint: str
    ambiguous_products: tuple[str, ...]


def _model_fields(value: Any) -> dict[str, Any]:
    """为流式 JSON 编码投影模型的声明字段，保留子对象身份，不执行 asdict 深拷贝。

    例如 Model() 返回 {}；内部复用凭据不属于声明字段，不进入摘要。
    非模型对象抛固定类型错误，不隐式转换未知对象或忽略非法字段值。
    """
    if not isinstance(value, Model):
        raise TypeError("research_event_parent_fingerprint_type_invalid")
    return {field.name: getattr(value, field.name) for field in fields(value)}


def parent_fingerprint(parent: ResearchResult) -> str:
    """逐块计算完整父结果的规范 JSON SHA256，不调用 to_json 或保留完整序列化副本。

    例如只改变父结果 details_json，输出摘要随之改变；输出等于 to_json 的 UTF8
    SHA256。每个唯一有效父结果只计算一次，额外编码内存受最大单字段大小约束。
    """
    checksum = hashlib.sha256()
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_model_fields,
    )
    for chunk in encoder.iterencode(parent):
        checksum.update(chunk.encode("utf-8"))
    return checksum.hexdigest()


def index_parents(
    inputs: ResearchInputs, upstream: Iterable[ResearchResult]
) -> dict[str, ParentBundle]:
    """一次遍历父输入，以 result_id 去重后按实体建立索引，不做全量两两比较。

    例如同证券的 P04/P06/P07 返回一个含三个父 ID 的 bundle；重复相同对象不序列化，
    同 ID 的不同对象才比较完整规范 JSON，冲突报 research_event_parent_conflict。
    错截止父结果不参与；缺实体报固定错误；每实体超过64个唯一父版本整组失败。
    多版本全部保留并标记歧义，父排序及摘要与输入顺序无关，也不受重复次数影响。
    """
    unique: dict[str, ResearchResult] = {}
    for parent in upstream:
        if not isinstance(parent.envelope.entity_id, str) or not parent.envelope.entity_id:
            raise ValueError("research_event_parent_entity_missing")
        result_id = parent.envelope.result_id
        previous = unique.get(result_id)
        if previous is not None:
            if previous is not parent and previous.to_json() != parent.to_json():
                raise ValueError("research_event_parent_conflict")
        else:
            unique[result_id] = parent
    grouped: dict[str, list[str]] = {}
    for result_id, parent in unique.items():
        if parent.envelope.analysis_as_of == inputs.analysis_as_of:
            grouped.setdefault(parent.envelope.entity_id, []).append(result_id)
    bundles = {}
    for identity, result_ids in grouped.items():
        if len(result_ids) > MAXIMUM_ENTITY_PARENTS:
            raise ValueError("research_event_parent_capacity_exceeded")
        ordered = tuple(sorted(result_ids))
        parents = tuple(unique[result_id] for result_id in ordered)
        products = Counter(parent.envelope.product_id for parent in parents)
        bundles[identity] = ParentBundle(
            parents=parents,
            result_ids=ordered,
            fingerprint=digest(
                [(result_id, parent_fingerprint(unique[result_id])) for result_id in ordered]
            ),
            ambiguous_products=tuple(sorted(name for name, count in products.items() if count > 1)),
        )
    return bundles
