"""以用户覆盖矩阵为不可静默扩展的 206 接口消费登记事实。"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from ..catalog import load_policies
from ..reference_contract import ReferenceContract, load_reference_contracts
from ..reference_storage import file_checksum

REGISTRY_VERSION = "product-consumers/v1"
EXPECTED_API_COUNT = 206
SOURCE_SHA256 = "8f493369a00af8ed461fb74330bd22f7bf1cb14c3a1702993ad82d9d2096bceb"
REGISTRY_PATH = Path(__file__).parents[1] / "data" / "product_consumers.csv"


@dataclass(frozen=True)
class ConsumerSpec:
    """接口主归属与选用字段；这里只登记消费计划，不声称业务产品已实现。"""

    api: str
    category: str
    product: str
    role: str
    purpose: str
    business_key: tuple[str, ...]
    selected_fields: tuple[str, ...]
    declared_units: dict[str, str]
    source_doc: str


def load_registry(path: Path | None = None) -> tuple[ConsumerSpec, ...]:
    """完全离线加载矩阵并按接口排序；显式路径不存在时不回退默认文件。

    输入 None，输出 206 项 ConsumerSpec；数组字段恢复为 tuple，单位恢复为字典。
    默认打包文件必须匹配原附件摘要，防止悄悄改写事实集合。
    """
    selected = REGISTRY_PATH if path is None else path
    if path is None and file_checksum(selected) != SOURCE_SHA256:
        raise ValueError("registry_source_digest_mismatch")
    with selected.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    entries = []
    required = {
        "api",
        "category",
        "product",
        "role",
        "purpose",
        "business_key",
        "selected_fields",
        "declared_units",
        "source_doc",
    }
    for row in rows:
        if set(row) != required or any(
            not isinstance(value, str) or not value.strip() for value in row.values()
        ):
            raise ValueError("registry_fields_invalid")
        try:
            units = json.loads(row["declared_units"])
        except ValueError:
            raise ValueError("registry_units_invalid") from None
        if not isinstance(units, dict) or any(
            type(key) is not str or type(value) is not str for key, value in units.items()
        ):
            raise ValueError("registry_units_invalid")
        keys = tuple(row["business_key"].split(","))
        selected_fields = tuple(row["selected_fields"].split(","))
        if any(not value.strip() for value in keys + selected_fields):
            raise ValueError("registry_fields_invalid")
        entries.append(
            ConsumerSpec(
                api=row["api"],
                category=row["category"],
                product=row["product"],
                role=row["role"],
                purpose=row["purpose"],
                business_key=keys,
                selected_fields=selected_fields,
                declared_units=units,
                source_doc=row["source_doc"],
            )
        )
    return tuple(sorted(entries, key=lambda item: item.api))


def validate_registry(
    entries: tuple[ConsumerSpec, ...],
    contracts: dict[str, ReferenceContract] | None = None,
) -> None:
    """验证精确附件集合、每项事实和当前字段契约，不用额外接口补数量。

    输入原矩阵和运行时契约，返回 None；缺一接口、换合法子集或主键漂移均报错。
    pro_bar 通过 load_reference_contracts 的 daily 组合契约完成校验。
    """
    expected = {item.api: item for item in load_registry()}
    if (
        len(entries) != EXPECTED_API_COUNT
        or len({item.api for item in entries}) != EXPECTED_API_COUNT
    ):
        raise ValueError("registry_api_count_mismatch")
    if {item.api for item in entries} != set(expected):
        raise ValueError("registry_api_set_mismatch")
    active = load_reference_contracts(load_policies()) if contracts is None else contracts
    for entry in entries:
        if entry != expected[entry.api]:
            raise ValueError("registry_source_fact_mismatch")
        contract = active.get(entry.api)
        if contract is None:
            raise ValueError("registry_contract_missing")
        fields = {item["name"] for item in contract.documentation.get("output_fields", [])}
        if (
            entry.category != contract.policy.category
            or entry.business_key != tuple(contract.primary_key)
            or not set(entry.selected_fields + entry.business_key) <= fields
            or entry.source_doc != contract.documentation.get("doc_url")
            or entry.declared_units != contract.policy.units
        ):
            raise ValueError("registry_contract_mismatch")
