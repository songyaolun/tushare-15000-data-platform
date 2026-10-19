"""真实字段结构的合成离线证据；不使用凭据或网络验证市场权限。"""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import pytest

from tushare_entitlement_hub.catalog import load_policies
from tushare_entitlement_hub.raw import RawStore, canonical_json
from tushare_entitlement_hub.reference_contract import load_reference_contracts
from tushare_entitlement_hub.regime_research.execution_market import (
    ExecutionArtifact,
    ExecutionDemand,
    audit_execution_market,
)


def sha(value):
    """返回字节SHA256；例如b''得到标准空消息摘要。"""
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def contracts():
    """复用真实接口契约并显式添加合成语义；输出两个仅用于fixture的契约。"""
    loaded = load_reference_contracts(load_policies())
    result = {}
    for api in ("stk_mins", "stk_auction_o"):
        contract = loaded[api]
        units = {field: "元" for field in ("open", "close", "high", "low")}
        units.update(vol="股", amount="元")
        result[api] = replace(contract, policy=replace(contract.policy, units=units), rules={
            "timezone": "Asia/Shanghai", "execution_semantics_evidence": "fixture-only",
            "execution_time_semantics": "bar_end_1min" if api == "stk_mins" else "opening_auction",
            "execution_window_start": "09:30:00" if api == "stk_mins" else "09:15:00",
            "execution_window_end": "09:31:00" if api == "stk_mins" else "09:25:00",
        })
    return result


def demand_for(state="trading", simulated_fill=True):
    """构建有哈希状态证据的需求；默认要求一个09:31分钟标签。"""
    payload = canonical_json({"ts_code": "000001.SZ", "date": "20240102", "state": state,
                              "source_reference": "synthetic-calendar-listing-suspension"}).encode()
    return ExecutionDemand("000001.SZ", "20240102", ("09:31:00",), simulated_fill,
                           payload, sha(payload))


def artifact_for(contract, items=None):
    """构造与RawStore相同的合成信封；默认一行10元、100股、1000元。"""
    api = contract.policy.api_name
    minute = api == "stk_mins"
    fields = ["ts_code", "trade_time" if minute else "trade_date",
              "open", "high", "low", "close", "vol", "amount"]
    row = ["000001.SZ", "2024-01-02 09:31:00" if minute else "20240102",
           10, 10, 10, 10, 100, 1000]
    parameters = {"ts_code": "000001.SZ"}
    if minute:
        parameters.update(freq="1min", start_date="2024-01-02 09:00:00",
                          end_date="2024-01-02 15:30:00")
    else:
        parameters["trade_date"] = "20240102"
    response = {"code": 0, "data": {"fields": fields,
                "items": [row] if items is None else items, "has_more": False}}
    envelope = {"api_name": api, "params": parameters, "response": response,
                "observed_at": "2026-10-20T00:00:00Z"}
    envelope["request_id"] = sha(canonical_json({"api_name": api, "params": parameters}).encode())
    envelope["response_hash"] = sha(canonical_json(response).encode())
    payload = canonical_json(envelope).encode()
    return ExecutionArtifact(api, "000001.SZ", "20240102", payload, sha(payload),
                             contract.fingerprint)


def changed(artifact, path, value, resign=True):
    """在指定路径替换合成字段并重新绑定哈希；例response.code=1测试供应商拒绝。"""
    envelope = json.loads(artifact.payload)
    target = envelope
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    if resign:
        envelope["request_id"] = sha(json.dumps({"api_name": artifact.api_name,
            "params": envelope["params"]}, sort_keys=True, separators=(",", ":")).encode())
        envelope["response_hash"] = sha(json.dumps(envelope["response"], sort_keys=True,
                                                   separators=(",", ":")).encode())
    payload = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
    return replace(artifact, payload=payload, content_sha256=sha(payload))


def test_complete_data_and_identical_duplicates_keep_capacity_unproven(contracts):
    """场景：两接口完整且重复一份分钟副本；预期覆盖1日但不认证PIT、权限或容量。"""
    artifacts = [artifact_for(contract) for contract in contracts.values()]
    result = audit_execution_market([demand_for()], artifacts + [artifacts[0]], contracts)
    assert result["passed"] and result["market_covered"] == 1
    assert result["results"][0]["rows_by_api"] == {"stk_mins": 1, "stk_auction_o": 1}
    assert not result["capacity_certified"] and not result["historical_pit_certified"]
    assert not result["current_permission_verified"]


def test_missing_data_preserves_full_denominator(contracts):
    """场景：需求无响应也无状态证据；预期保留两条证券日与全部缺口。"""
    demands = [ExecutionDemand("000001.SZ", "20240102"), ExecutionDemand("600000.SH", "20240102")]
    result = audit_execution_market(demands, [], contracts)
    assert result["required_security_days"] == result["unresolved"] == 2
    assert result["market_covered"] == 0
    assert "session_state_unknown" in result["results"][0]["errors"]
    assert not audit_execution_market([], [], contracts)["passed"]


@pytest.mark.parametrize("state", ["suspended", "not_listed", "non_trading_day"])
def test_independent_absence_evidence_requires_no_simulated_fill(contracts, state):
    """场景：停牌、未上市或休市有状态证据；预期只确认无数据解释，存在成交则拒绝。"""
    result = audit_execution_market([demand_for(state, False)], [], contracts)
    assert result["passed"] and result["absence_evidenced"] == 1
    assert result["market_covered"] == 0
    contradiction = audit_execution_market([demand_for(state)], [], contracts)
    assert "absence_conflicts_with_fill" in contradiction["results"][0]["errors"]


def test_absence_cannot_hide_observed_market_rows(contracts):
    """场景：停牌声明与实际响应行冲突；预期拒绝而不删除响应。"""
    artifacts = [artifact_for(contract) for contract in contracts.values()]
    result = audit_execution_market([demand_for("suspended", False)], artifacts, contracts)
    assert "stk_mins:absence_conflicts_with_rows" in result["results"][0]["errors"]
    empty = [artifact_for(contract, []) for contract in contracts.values()]
    assert audit_execution_market([demand_for("suspended", False)], empty, contracts)["passed"]


@pytest.mark.parametrize("field,value,error", [
    ("ts_code", "600000.SH", "session_evidence_mismatch"),
    ("date", "20240103", "session_evidence_mismatch"),
    ("source_reference", "", "session_evidence_mismatch"),
    ("source_reference", None, "session_evidence_mismatch"),
    ("state", "unknown", "unknown_session_state"),
    ("state", [], "unknown_session_state"),
])
def test_session_evidence_is_bound_to_security_date(contracts, field, value, error):
    """场景：独立状态证据字段不完整或错证券日；预期固定错误且不确认无数据解释。"""
    demand = demand_for()
    evidence = json.loads(demand.session_evidence)
    evidence[field] = value
    payload = canonical_json(evidence).encode()
    demand = replace(demand, session_evidence=payload, session_sha256=sha(payload))
    assert error in audit_execution_market([demand], [], contracts)["results"][0]["errors"]


@pytest.mark.parametrize("payload", [b"[]", b"not-json", b'\xff', b'{"token":"fixture"}'])
def test_invalid_or_private_state_json_is_redacted(contracts, payload):
    """场景：非法JSON、非对象、非法编码或凭据字段；预期固定错误而不回显内容。"""
    demand = replace(demand_for(), session_evidence=payload, session_sha256=sha(payload))
    result = audit_execution_market([demand], [], contracts)
    assert "invalid_public_json" in result["results"][0]["errors"]
    assert "fixture" not in str(result)


def test_unknown_native_contracts_fail_closed():
    """场景：直接使用生产契约的未知单位和时段；预期保留未知，不按样例猜测。"""
    contracts = load_reference_contracts(load_policies())
    artifacts = [artifact_for(contracts[api]) for api in ("stk_mins", "stk_auction_o")]
    result = audit_execution_market([demand_for()], artifacts, contracts)
    assert "stk_auction_o:unit_unknown" in result["results"][0]["errors"]
    assert "stk_auction_o:timezone_unknown" in result["results"][0]["errors"]
    assert "stk_mins:time_semantics_unknown" in result["results"][0]["errors"]


@pytest.mark.parametrize("path,value,error,resign", [
    (("api_name",), "rt_min", "invalid_envelope", True),
    (("params",), [], "invalid_envelope", True),
    (("response",), [], "invalid_envelope", True),
    (("request_id",), "wrong", "envelope_hash_mismatch", False),
    (("response_hash",), "wrong", "envelope_hash_mismatch", False),
    (("observed_at",), None, "observation_time_invalid", True),
    (("observed_at",), "2024-01-02", "observation_time_invalid", True),
    (("observed_at",), "2024-01-02T01:30:00Z", "row_time_invalid", True),
    (("response", "code"), True, "provider_response_failed", True),
    (("response", "code"), 1, "provider_response_failed", True),
    (("params", "ts_code"), "600000.SH", "request_scope_mismatch", True),
    (("params", "freq"), "5min", "frequency_mismatch", True),
    (("params", "start_date"), "2024-01-03 09:00:00", "request_window_invalid", True),
    (("params", "end_date"), "2024-01-03 15:30:00", "request_window_invalid", True),
    (("params", "end_date"), "2024-01-02 08:00:00", "request_window_invalid", True),
    (("params", "start_date"), None, "request_window_invalid", True),
    (("response", "data"), None, "response_shape_invalid", True),
    (("response", "data", "fields"), None, "response_shape_invalid", True),
    (("response", "data", "fields"), [1], "response_shape_invalid", True),
    (("response", "data", "fields"), ["ts_code", "ts_code"], "response_shape_invalid", True),
    (("response", "data", "items"), None, "response_shape_invalid", True),
    (("response", "data", "has_more"), True, "response_truncated", True),
    (("response", "data", "fields"), ["ts_code"], "required_fields_missing", True),
    (("response", "data", "items"), [[1]], "row_shape_invalid", True),
    (("response", "data", "items"), [{}], "row_shape_invalid", True),
    (("response", "data", "items", 0, 0), "600000.SH", "row_scope_mismatch", True),
    (("response", "data", "items", 0, 1), "2024-01-03 09:31:00", "row_time_invalid", True),
    (("response", "data", "items", 0, 1), "2024-01-02 09:31:01", "row_time_invalid", True),
    (("response", "data", "items", 0, 1), "2024-01-02 09:31:00.001", "row_time_invalid", True),
    (("response", "data", "items", 0, 1), "2024-01-02 08:59:00", "row_time_invalid", True),
    (("response", "data", "items", 0, 1), None, "row_time_invalid", True),
    (("response", "data", "items", 0, 2), True, "row_values_invalid", True),
    (("response", "data", "items", 0, 2), None, "row_values_invalid", True),
    (("response", "data", "items", 0, 2), "inf", "row_values_invalid", True),
    (("response", "data", "items", 0, 2), 0, "row_values_invalid", True),
    (("response", "data", "items", 0, 2), 11, "row_values_invalid", True),
    (("response", "data", "items", 0, 5), 9, "row_values_invalid", True),
    (("response", "data", "items", 0, 6), -1, "row_values_invalid", True),
    (("response", "data", "items", 0, 6), 0, "row_values_invalid", True),
    (("response", "data", "items", 0, 7), 1200, "row_values_invalid", True),
])
def test_corrupt_minute_evidence_is_rejected(contracts, path, value, error, resign):
    """场景：逐项破坏原始证据或量价契约；预期匹配具体拒绝原因且覆盖仍为0。"""
    artifact = changed(artifact_for(contracts["stk_mins"]), path, value, resign)
    result = audit_execution_market([demand_for()], [artifact], contracts)
    assert "stk_mins:" + error in result["results"][0]["errors"]
    assert result["market_covered"] == 0


@pytest.mark.parametrize("path,value,error", [
    (("params", "trade_date"), "20240103", "request_scope_mismatch"),
    (("response", "data", "items", 0, 1), "20240103", "row_scope_mismatch"),
    (("response", "data", "items"), [], "auction_coverage_incomplete"),
    (("observed_at",), "2024-01-01T00:00:00Z", "observation_before_session"),
    (("observed_at",), "2024-01-02T01:24:00Z", "observation_before_execution_window"),
])
def test_auction_scope_and_empty_response_fail(contracts, path, value, error):
    """场景：竞价请求、响应日期不符或正常交易日为空；预期拒绝该竞价证据。"""
    artifact = changed(artifact_for(contracts["stk_auction_o"]), path, value)
    result = audit_execution_market([demand_for()], [artifact], contracts)
    assert "stk_auction_o:" + error in result["results"][0]["errors"]


def test_hashes_contract_presence_and_unexpected_artifacts(contracts):
    """场景：内容哈希、合同哈希、合同键或请求范围不符；预期各自拒绝且不缩减分母。"""
    artifact = artifact_for(contracts["stk_mins"])
    for invalid, error in [(replace(artifact, content_sha256="wrong"), "content_hash_mismatch"),
                           (replace(artifact, contract_hash="wrong"), "contract_hash_mismatch")]:
        result = audit_execution_market([demand_for()], [invalid], contracts)
        assert "stk_mins:" + error in result["results"][0]["errors"]
    for missing in [{}, {"stk_mins": contracts["stk_auction_o"]}]:
        assert "stk_mins:contract_missing" in audit_execution_market(
            [demand_for()], [artifact], missing)["results"][0]["errors"]
    unknown = [replace(artifact, api_name="rt_min"), replace(artifact, ts_code="600000.SH")]
    assert audit_execution_market([demand_for()], unknown, contracts)["unexpected_artifacts"] == 2
    bad_state = replace(demand_for(), session_sha256="wrong")
    assert "content_hash_mismatch" in audit_execution_market(
        [bad_state], [], contracts)["results"][0]["errors"]


def test_conflicting_duplicates_and_missing_minutes(contracts):
    """场景：重复分钟量额不同或时间表未知／缺行；预期拒绝并保留证券日。"""
    artifact = artifact_for(contracts["stk_mins"])
    conflict = changed(artifact, ("response", "data", "items", 0, 6), 200)
    conflict = changed(conflict, ("response", "data", "items", 0, 7), 2000)
    result = audit_execution_market([demand_for()], [artifact, conflict], contracts)
    assert "stk_mins:duplicate_conflict" in result["results"][0]["errors"]
    for labels, error in [((), "minute_schedule_unknown"),
                          (("09:31:00", "09:32:00"), "minute_coverage_incomplete")]:
        result = audit_execution_market([replace(demand_for(), expected_minutes=labels)],
                                        [artifact], contracts)
        assert "stk_mins:" + error in result["results"][0]["errors"]


def test_aware_timestamp_and_zero_activity_bars(contracts):
    """场景：显式UTC分钟及零量额参考价格；预期按上海时间验证而不制造成交量。"""
    minute = artifact_for(contracts["stk_mins"])
    minute = changed(minute, ("response", "data", "items", 0, 1), "2024-01-02T01:31:00+00:00")
    minute = changed(minute, ("response", "data", "items", 0, 6), 0)
    minute = changed(minute, ("response", "data", "items", 0, 7), 0)
    result = audit_execution_market(
        [demand_for()], [minute, artifact_for(contracts["stk_auction_o"])], contracts)
    assert result["passed"] and not result["capacity_certified"]


def test_documented_row_cap_and_missing_semantic_reference(contracts):
    """场景：响应碰到行数上限或合同缺区间证据；预期不能以少量有效行算完整覆盖。"""
    minute_contract = contracts["stk_mins"]
    limited = replace(minute_contract, policy=replace(minute_contract.policy, page_size=1))
    result = audit_execution_market([demand_for()], [artifact_for(limited)], {"stk_mins": limited})
    assert "stk_mins:response_truncated" in result["results"][0]["errors"]
    invalid = replace(minute_contract, policy=replace(minute_contract.policy, page_size=0),
                      rules={**minute_contract.rules, "execution_semantics_evidence": ""})
    result = audit_execution_market([demand_for()], [artifact_for(invalid)], {"stk_mins": invalid})
    assert "stk_mins:row_limit_unknown" in result["results"][0]["errors"]
    assert "stk_mins:time_semantics_unknown" in result["results"][0]["errors"]


@pytest.mark.parametrize("start,end", [("09:26:00", "09:25:00"),
                                       ("9:15:00", "09:25:00"),
                                       ("09:15:00", "9:25:00")])
def test_execution_interval_must_be_explicit_and_in_order(contracts, start, end):
    """场景：只有开盘竞价标签却没有有效起止区间；预期不能认证时间语义。"""
    contract = contracts["stk_auction_o"]
    contract = replace(contract, rules={**contract.rules, "execution_window_start": start,
                                       "execution_window_end": end})
    result = audit_execution_market([demand_for()], [artifact_for(contract)],
                                    {"stk_auction_o": contract})
    assert "stk_auction_o:execution_window_unknown" in result["results"][0]["errors"]
    minute = contracts["stk_mins"]
    minute = replace(minute, rules={**minute.rules, "execution_window_start": "09:32:00",
                                    "execution_window_end": "09:33:00"})
    result = audit_execution_market([demand_for()], [artifact_for(minute)], {"stk_mins": minute})
    assert "stk_mins:row_outside_execution_window" in result["results"][0]["errors"]


@pytest.mark.parametrize("changes,error", [
    ({"date": "2024-01-02"}, "demand_date_invalid"),
    ({"date": "bad"}, "demand_date_invalid"),
    ({"expected_minutes": ("9:31:00",)}, "minute_schedule_invalid"),
    ({"expected_minutes": ("bad",)}, "minute_schedule_invalid"),
    ({"expected_minutes": ("09:31:00", "09:31:00")}, "minute_schedule_duplicate"),
])
def test_invalid_demand_metadata_is_rejected(contracts, changes, error):
    """场景：需求日期或分钟标签非法；预期固定错误，不能悄悄归一化分母。"""
    with pytest.raises(ValueError, match=error):
        audit_execution_market([replace(demand_for(), **changes)], [], contracts)


def test_duplicate_demands_fail_before_processing(contracts):
    """场景：同证券日重复登记；预期拒绝避免重复计算覆盖分母。"""
    with pytest.raises(ValueError, match="duplicate_demand"):
        audit_execution_market([demand_for(), demand_for()], [], contracts)


def test_inputs_are_not_mutated(contracts):
    """场景：审计完成后回读输入；预期需求、原始字节和契约均保持原值。"""
    demand = demand_for()
    artifact = artifact_for(contracts["stk_mins"])
    before = deepcopy((demand, artifact, contracts))
    audit_execution_market([demand], [artifact], contracts)
    assert (demand, artifact, contracts) == before


def test_existing_raw_store_outputs_are_accepted_without_repacking(contracts, tmp_path):
    """场景：既有RawStore真实写入离线响应；预期原字节可直接核验且不需要第二套信封。"""
    artifacts = []
    for contract in contracts.values():
        template = artifact_for(contract)
        envelope = json.loads(template.payload)
        record = RawStore(tmp_path).write(contract.policy.api_name, envelope["params"],
                                          envelope["response"], envelope["observed_at"])
        payload = record.path.read_bytes()
        artifacts.append(replace(template, payload=payload, content_sha256=sha(payload)))
    assert audit_execution_market([demand_for()], artifacts, contracts)["passed"]
