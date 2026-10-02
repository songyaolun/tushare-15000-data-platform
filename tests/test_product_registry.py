"""公共产品消费注册表的确定性加载、事实集合及字段契约校验。"""

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from tushare_entitlement_hub.catalog import load_policies
from tushare_entitlement_hub.product_foundation.registry import (
    load_registry,
    validate_registry,
)
from tushare_entitlement_hub.reference_contract import load_reference_contracts


def test_load_registry_preserves_complete_packaged_facts():
    """场景：默认加载固定矩阵；预期：206 个唯一接口和19类事实逐字段保留。

    输入：默认 CSV；输出示例：stock_basic 的业务键为 ('ts_code',)。
    """
    entries = load_registry()
    assert isinstance(entries, tuple)
    assert len(entries) == len({entry.api for entry in entries}) == 206
    assert len({entry.category for entry in entries}) == 19
    assert load_registry() == entries
    source = Path(__file__).parents[1] / "src/tushare_entitlement_hub/data/product_consumers.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        original = list(csv.DictReader(handle))
    expected = []
    for row in original:
        expected.append(
            {
                **row,
                "business_key": tuple(row["business_key"].split(",")),
                "selected_fields": tuple(row["selected_fields"].split(",")),
                "declared_units": json.loads(row["declared_units"]),
            }
        )
    assert sorted((asdict(entry) for entry in entries), key=lambda item: item["api"]) == sorted(
        expected, key=lambda item: item["api"]
    )
    stock = next(entry for entry in entries if entry.api == "stock_basic")
    assert stock.business_key == ("ts_code",)
    assert stock.product == "P03"
    assert stock.category == "basic"
    assert "list_date" in stock.selected_fields
    assert stock.source_doc == "https://tushare.pro/document/2?doc_id=25"
    assert validate_registry(entries) is None


def test_load_registry_explicit_copy_matches_default(tmp_path):
    """场景：显式传入离线矩阵副本；预期：解析结果与默认路径一致。

    输入：tmp_path 下相同 CSV；输出：与 load_registry() 相同的 tuple。
    """
    source = Path(__file__).parents[1] / "src/tushare_entitlement_hub/data/product_consumers.csv"
    copied = tmp_path / "consumers.csv"
    copied.write_bytes(source.read_bytes())
    assert load_registry(copied) == load_registry()


@pytest.mark.parametrize("mode", ["empty", "missing", "duplicate", "unknown"])
def test_validate_registry_rejects_incomplete_or_replaced_interface_set(mode):
    """场景：矩阵缺失、重复或用未知接口替换；预期：不能按数量冒充完整集合。

    输入示例：206 行中最后一行重复第一行；输出：有原因的 ValueError。
    """
    entries = load_registry()
    candidates = {
        "empty": (),
        "missing": entries[:-1],
        "duplicate": entries[:-1] + (entries[0],),
        "unknown": entries[:-1] + (replace(entries[-1], api="unknown_product_api"),),
    }
    with pytest.raises(ValueError) as error:
        validate_registry(candidates[mode])
    assert str(error.value)


@pytest.mark.parametrize(
    "changes",
    [
        {"category": "unknown_category"},
        {"product": "P99"},
        {"role": ""},
        {"purpose": ""},
        {"business_key": ("ts_code",)},
        {"selected_fields": ("does_not_exist",)},
        {"selected_fields": ("pe",)},
        {"declared_units": {"pe": "元"}},
        {"source_doc": "https://tushare.pro/document/2?doc_id=25"},
    ],
    ids=[
        "category",
        "product",
        "role",
        "purpose",
        "key",
        "unknown-field",
        "subset",
        "units",
        "doc",
    ],
)
def test_validate_registry_rejects_changed_csv_facts(changes):
    """场景：只改一项已登记事实；预期：字段合法也不能悄悄偏离固定附件。

    输入示例：bak_basic selected_fields 缩为 ('pe',)；输出：ValueError。
    """
    entries = load_registry()
    index = next(index for index, entry in enumerate(entries) if entry.api == "bak_basic")
    altered = list(entries)
    altered[index] = replace(entries[index], **changes)
    with pytest.raises(ValueError) as error:
        validate_registry(tuple(altered))
    assert str(error.value)


def test_validate_registry_uses_runtime_contracts_and_pro_bar_composition():
    """场景：运行时契约与 SDK 组合接口；预期：pro_bar 复用 daily 字段且校验通过。

    输入：本地政策及字段契约；输出：206 行合法，pro_bar 字段来自 daily。
    """
    contracts = load_reference_contracts(load_policies())
    entries = load_registry()
    pro_bar = next(entry for entry in entries if entry.api == "pro_bar")
    daily_fields = {field["name"] for field in contracts["daily"].documentation["output_fields"]}
    assert set(pro_bar.selected_fields) <= daily_fields
    assert contracts["pro_bar"].documentation["schema_basis"] == (
        "daily_fields_used_by_https_composition"
    )
    assert validate_registry(entries, contracts=contracts) is None


@pytest.mark.parametrize("change", ["empty", "missing", "field", "key", "category", "source"])
def test_validate_registry_rejects_runtime_contract_drift(change):
    """场景：CSV 未变但来源契约漂移；预期：缺契约、字段、主键、类别或文档不一致均拒绝。

    输入示例：daily 契约不再声明 close；输出：有原因的 ValueError。
    """
    contracts = load_reference_contracts(load_policies())
    contract = contracts["daily"]
    if change == "empty":
        contracts.clear()
    elif change == "missing":
        contracts.pop("daily")
    elif change == "field":
        contract.documentation = {
            **contract.documentation,
            "output_fields": [
                field
                for field in contract.documentation["output_fields"]
                if field["name"] != "close"
            ],
        }
    elif change == "key":
        contract.rules = {**contract.rules, "primary_key": ["ts_code"]}
    elif change == "category":
        contract.policy = replace(contract.policy, category="unknown_category")
    else:
        contract.documentation = {**contract.documentation, "doc_url": "https://example.invalid"}
    with pytest.raises(ValueError) as error:
        validate_registry(load_registry(), contracts=contracts)
    assert str(error.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("api", ""),
        ("business_key", ""),
        ("selected_fields", ""),
        ("selected_fields", "pe,"),
        ("declared_units", "not-json"),
        ("declared_units", "[]"),
        ("declared_units", '{"close": 1000}'),
    ],
)
def test_load_registry_rejects_invalid_csv_values(tmp_path, field, value):
    """场景：输入 CSV 的身份、键、字段或单位映射损坏；预期：加载边界明确拒绝。

    输入示例：declared_units='[]'；输出：ValueError，不产生部分注册表。
    """
    source = Path(__file__).parents[1] / "src/tushare_entitlement_hub/data/product_consumers.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0][field] = value
    corrupted = tmp_path / "invalid.csv"
    with corrupted.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError) as error:
        load_registry(corrupted)
    assert str(error.value)


def test_load_registry_missing_file_preserves_filesystem_error(tmp_path):
    """场景：调用方明确指定不存在的矩阵；预期：报文件不存在，不悄悄回退默认数据。

    输入：missing.csv；输出：FileNotFoundError。
    """
    with pytest.raises(FileNotFoundError):
        load_registry(tmp_path / "missing.csv")


def test_load_registry_rejects_tampered_packaged_source(tmp_path, monkeypatch):
    """场景：默认打包来源被改写；预期：摘要校验失败，不能发布新事实冒充附件。

    输入：将默认来源路径指向改写副本；输出：registry_source_digest_mismatch。
    """
    from tushare_entitlement_hub.product_foundation import registry

    changed = tmp_path / "tampered.csv"
    changed.write_text("category,api\nbasic,invented\n", encoding="utf-8")
    monkeypatch.setattr(registry, "REGISTRY_PATH", changed)
    with pytest.raises(ValueError, match="registry_source_digest_mismatch"):
        load_registry()
