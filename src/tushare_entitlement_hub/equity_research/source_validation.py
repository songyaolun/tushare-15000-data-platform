"""将当前研究证据批量回查只读accepted行，核对三层身份和字段定位。"""

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import duckdb

from ..catalog import load_policies
from ..product_foundation.contracts import EvidenceRef
from ..product_foundation.numeric import source_evidence
from ..product_foundation.reference_scan import map_availability
from ..reference_contract import load_reference_contracts
from .models import ResearchConfig


def verify_source_evidence(
    reference_path: Path,
    products: duckdb.DuckDBPyConnection,
    run_id: str,
    config: ResearchConfig,
    *,
    batch_size: int = 512,
) -> None:
    """逐批核对已用与反证引用，拒绝来源不存在、隔离、契约或字段身份漂移。

    例：把证据source_record_id改成不存在的记录，即使产品内子表同步修改也拒绝。
    仅校验当前参考指纹对应批次；其他历史批次仍校验其内部模型及完整血缘。
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("invalid_batch_size")
    contracts = load_reference_contracts(load_policies())
    mappings = {mapping.source_api: mapping for mapping in config.foundation.time_mappings}
    assertion_groups = [item.evidence_refs for item in config.foundation.aliases]
    assertion_groups.extend(item.evidence_refs for item in config.foundation.certifications)
    assertion_groups.extend(item.evidence_refs for item in config.confirmed_relations)
    assertions = {reference.to_json() for group in assertion_groups for reference in group}
    with duckdb.connect(
        str(reference_path), read_only=True, config={"threads": 1, "memory_limit": "512MB"}
    ) as source:
        cursor = products.execute(
            "SELECT DISTINCT payload FROM research.evidence WHERE run_id=?", [run_id]
        )
        while batch := cursor.fetchmany(batch_size):
            grouped: dict[str, list[EvidenceRef]] = defaultdict(list)
            for (payload,) in batch:
                reference = EvidenceRef.from_json(payload)
                if reference.to_json() not in assertions:
                    grouped[reference.source_api].append(reference)
            for api, references in grouped.items():
                if api not in contracts or api not in mappings:
                    raise ValueError("research_evidence_source_unknown")
                # API来自已加载契约白名单，证券ID使用参数绑定。
                result = source.execute(
                    f'SELECT to_json(source_row) FROM accepted."{api}" source_row '
                    "WHERE source_record_id IN (SELECT unnest(?))",
                    [sorted({reference.source_record_id for reference in references})],
                ).fetchall()
                records: dict[str, dict[str, Any]] = {}
                for (payload,) in result:
                    row = json.loads(payload)
                    record_id = row["source_record_id"]
                    if record_id in records:
                        raise ValueError("research_evidence_source_duplicate")
                    occurrences = row.get("source_occurrences")
                    if isinstance(occurrences, str):
                        row["source_occurrences"] = json.loads(occurrences)
                    records[record_id] = row
                for reference in references:
                    row = records.get(reference.source_record_id)
                    if (
                        row is None
                        or row.get("quality_tier") not in {"verified", "usable_with_annotation"}
                        or row.get("is_quarantined") is True
                    ):
                        raise ValueError("research_evidence_source_missing")
                    if any(field not in row for field in reference.fields):
                        raise ValueError("research_evidence_field_missing")
                    if (
                        row.get("contract_hash") != contracts[api].fingerprint
                        or row.get("source_api") != api
                    ):
                        raise ValueError("research_evidence_contract_mismatch")
                    availability = map_availability(row, contracts[api], mappings[api])
                    expected = replace(
                        source_evidence(row, reference.fields[0], availability),
                        fields=reference.fields,
                    )
                    if expected != reference:
                        raise ValueError("research_evidence_identity_mismatch")
