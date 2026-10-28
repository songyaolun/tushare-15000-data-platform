"""传输、分页与原始数据层的离线契约测试，所有访问凭据均为固定占位符。"""

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from tushare_entitlement_hub import raw as raw_module
from tushare_entitlement_hub.client import (
    Client,
    FakeTransport,
    HttpTransport,
    Limiter,
    ProviderError,
    Transport,
    _contains_secret,
)
from tushare_entitlement_hub.raw import RawRecord, RawStore, assert_public, canonical_json

OBSERVED = "2024-01-03T00:00:00Z"
PLACEHOLDER = "<REDACTED>"


def response(items=None, fields=None):
    """构造确定性公开响应；输入 [[1]]，输出字段 id 和一行数据的成功响应。"""
    return {"code": 0, "data": {
        "fields": ["id"] if fields is None else fields,
        "items": [] if items is None else items,
    }}


def client_for(tmp_path, responses, **kwargs):
    """创建离线客户端；输入响应列表，输出不会联网且不进行真实等待的 Client。"""
    return Client(FakeTransport(responses), RawStore(tmp_path), **kwargs)


@pytest.mark.parametrize("first", [None, ""])
def test_forecast_nullable_date_keeps_raw_identity_across_pages(tmp_path, first):
    """同公告日空首次日期及更晚首次日期跨页保留，原响应和值均不填补。"""
    fields = ["ts_code", "end_date", "ann_date", "first_ann_date"]
    rows = [["000001.SZ", "20231231", "20240320", first],
            ["000001.SZ", "20231231", "20240320", "20240420"]]
    responses = [response([row], fields) for row in rows] + [response([], fields)]
    client = client_for(tmp_path, responses)
    pages = client.pages("forecast_vip", {}, OBSERVED, fields, 1, 3)
    assert [page.rows[0]["first_ann_date"] for page in pages[:-1]] == [first, "20240420"]
    assert pages[0].raw.response["data"]["items"][0][-1] == first


@pytest.mark.parametrize("api, missing", [("daily", "first_ann_date"), ("forecast_vip", "ts_code")])
def test_nullable_forecast_date_does_not_relax_other_required_keys(tmp_path, api, missing):
    """只有预告首次日期可空；daily同名键或预告证券代码为空均失败。"""
    fields = ["ts_code", "first_ann_date"]
    row = {"ts_code": "000001.SZ", "first_ann_date": "20240320", missing: None}
    client = client_for(tmp_path, [response([[row[name] for name in fields]], fields)])
    with pytest.raises(ProviderError, match="主键为空"):
        client.pages(api, {}, OBSERVED, fields, 2, 1)


def test_nullable_forecast_date_rejects_duplicate_record(tmp_path):
    """同证券报告期公告日及空首次日期完全相同的跨页记录仍拒绝重复。"""
    fields = ["ts_code", "end_date", "ann_date", "first_ann_date"]
    row = ["000001.SZ", "20231231", "20240320", None]
    client = client_for(tmp_path, [response([row], fields)] * 2)
    with pytest.raises(ProviderError, match="主键重复"):
        client.pages("forecast_vip", {}, OBSERVED, fields, 1, 3)


@pytest.mark.parametrize("value", [
    {1: "public"}, {"nested": {"access_token": PLACEHOLDER}},
    {"fields": ["client_secret"]}, {"msg": "token=<REDACTED>"},
    {"COOKIE": PLACEHOLDER}, {"certificate": PLACEHOLDER},
])
def test_public_boundary_rejects_credential_shapes(value):
    """场景：嵌套或表格字段含凭据；预期：固定安全错误。例：token 字段→ValueError。"""
    with pytest.raises(ValueError) as caught:
        assert_public(value)
    assert PLACEHOLDER not in str(caught.value)


def test_canonical_json_is_stable_and_public():
    """场景：不同键顺序和嵌套公开值；预期：稳定 JSON。例：b、a→a、b 排序。"""
    assert canonical_json({"b": 2, "a": [1, None, True]}) == '{"a":[1,null,true],"b":2}'
    assert assert_public(({"fields": ["id", 1]}, "公开数据", 1)) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object(), {"a": {1, 2}}])
def test_canonical_json_rejects_non_json_without_exception_context(value):
    """场景：非有限数或不支持对象；预期：无原异常链。例：NaN→固定 ValueError。"""
    with pytest.raises(ValueError) as caught:
        canonical_json(value)
    assert caught.value.__context__ is None


def test_raw_store_idempotency_observations_and_readback(tmp_path):
    """场景：重放和后续观测；预期：请求号稳定、时间分别存档。例：两时刻→两文件。"""
    raw_store = RawStore(tmp_path)
    payload = response([[1]])
    first = raw_store.write("daily", {"b": 2, "a": 1}, payload, OBSERVED)
    replay = raw_store.write("daily", {"a": 1, "b": 2}, payload, OBSERVED)
    later = raw_store.write("daily", {"a": 1, "b": 2}, payload, "2024-01-04T00:00:00Z")
    assert first == replay
    assert first.path != later.path
    assert first.request_id == later.request_id
    assert first.response_hash == hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    stored = json.loads(first.path.read_text())
    assert stored["response"] == payload
    assert stored["params"] == {"a": 1, "b": 2}
    assert stored["observed_at"] == OBSERVED
    payload["data"]["items"].append([2])
    assert first.response["data"]["items"] == [[1]]
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("api_name,observed,payload", [
    ("../daily", OBSERVED, {"code": 0}),
    ("daily", "", {"code": 0}),
    ("daily", OBSERVED, {"code": 1}),
    ("daily", OBSERVED, {"code": False}),
])
def test_raw_invalid_metadata_never_creates_files(tmp_path, api_name, observed, payload):
    """场景：路径穿越、缺时间或失败状态；预期：写入前拒绝。例：code=1→零文件。"""
    with pytest.raises(ValueError):
        RawStore(tmp_path).write(api_name, {}, payload, observed)
    assert not list(tmp_path.rglob("*.json"))


def test_raw_store_detects_corruption_without_overwriting(tmp_path):
    """场景：已有哈希文件被改写；预期：校验失败且不覆盖证据。例：坏字节→拒绝。"""
    store = RawStore(tmp_path)
    raw = store.write("daily", {}, response(), OBSERVED)
    raw.path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="读回校验"):
        store.write("daily", {}, response(), OBSERVED)
    assert raw.path.read_bytes() == b"corrupted"


def test_raw_store_independent_response_hash_failure(tmp_path, monkeypatch):
    """场景：读回阶段哈希依赖故障；预期：独立响应校验拒绝。例：第五次摘要错误→失败。"""
    real_sha = hashlib.sha256
    calls = []

    def fail_readback_hash(data):
        """仅模拟读回响应的摘要计算故障；输入字节，第五次输出错误摘要对象。"""
        calls.append(data)
        return Mock(hexdigest=lambda: "invalid") if len(calls) == 5 else real_sha(data)

    monkeypatch.setattr(raw_module.hashlib, "sha256", fail_readback_hash)
    with pytest.raises(ValueError, match="内容哈希"):
        RawStore(tmp_path).write("daily", {}, response(), OBSERVED)
    assert len(calls) == 5


def test_raw_atomic_write_cleans_temporary_on_filesystem_failure(tmp_path, monkeypatch):
    """场景：原子替换发生磁盘错误；预期：临时文件清理。例：replace 抛 OSError→零临时。"""
    monkeypatch.setattr(raw_module.os, "replace", Mock(side_effect=OSError("磁盘故障")))
    with pytest.raises(OSError):
        RawStore(tmp_path).write("daily", {}, response(), OBSERVED)
    assert not list(tmp_path.rglob("*.tmp"))
    assert not list(tmp_path.rglob("*.json"))


def test_protocol_stub_does_not_send_a_request():
    """场景：调用协议默认声明；预期：不执行网络操作。例：协议方法→None。"""
    assert Transport.request(object(), "daily", {}) is None


@pytest.mark.parametrize("value,expected", [
    ({"echo": PLACEHOLDER}, True), ({PLACEHOLDER: 1}, True),
    ([0, (False, PLACEHOLDER)], True), ({"a": [1, None]}, False),
])
def test_secret_echo_detection_checks_keys_and_values(value, expected):
    """场景：凭据出现在多种嵌套位置；预期：内存检测准确。例：键回显→True。"""
    assert _contains_secret(value, PLACEHOLDER) is expected


@pytest.mark.parametrize("kwargs", [
    {"base_url": "http://api.tushare.pro"},
    {"base_url": "https://example.invalid"},
    {"base_url": "https://api.tushare.pro?redirect=1"},
    {"timeout": 0}, {"timeout": float("nan")},
])
def test_http_requires_official_https_and_positive_timeout(kwargs):
    """场景：非官方地址或错误超时；预期：构造即拒绝。例：http 地址→contract_failure。"""
    with pytest.raises(ProviderError) as caught:
        HttpTransport(lambda: PLACEHOLDER, **kwargs)
    assert caught.value.kind == "contract_failure"


def test_http_live_optin_precedes_credential_lookup():
    """场景：未显式开启真实访问；预期：不读取凭据。例：默认 request→unauthorized。"""
    provider = Mock(return_value=PLACEHOLDER)
    with HttpTransport(provider) as transport, pytest.raises(ProviderError) as caught:
        transport.request("daily", {})
    assert caught.value.kind == "unauthorized"
    provider.assert_not_called()


@pytest.mark.parametrize("provider", [
    Mock(return_value=""), Mock(side_effect=ValueError(PLACEHOLDER)),
])
def test_http_missing_credential_discards_original_exception(provider):
    """场景：凭据回调缺失或抛错；预期：无回调异常链。例：回调抛错→安全 unauthorized。"""
    with HttpTransport(provider, live=True) as transport, pytest.raises(ProviderError) as caught:
        transport.request("daily", {})
    assert caught.value.kind == "unauthorized"
    assert caught.value.__context__ is None
    assert PLACEHOLDER not in str(caught.value)


def test_http_reuses_client_sends_post_and_clears_cookies():
    """场景：连续两次离线 HTTP；预期：POST、末刻取凭据且不留 Cookie。例：两次→两请求。"""
    calls = []
    provider = Mock(return_value=PLACEHOLDER)

    def serve(request):
        """验证公开参数与占位凭据的发送格式；输入 POST，输出确定性成功响应。"""
        body = json.loads(request.content)
        assert request.method == "POST"
        assert str(request.url) == "https://api.tushare.pro"
        assert body == {"api_name": "daily", "token": PLACEHOLDER,
                        "params": {"a": 1}, "fields": ""}
        assert "cookie" not in request.headers
        calls.append(body["api_name"])
        return httpx.Response(200, json=response(), headers={"set-cookie": "session=public"})

    with HttpTransport(provider, live=True, http_transport=httpx.MockTransport(serve)) as transport:
        connection_pool = transport._client
        assert provider.call_count == 0
        assert transport.request("daily", {"a": 1}) == response()
        assert transport.request("daily", {"a": 1}) == response()
        assert transport._client is connection_pool
        assert not connection_pool.cookies
        assert connection_pool.trust_env is False
    assert connection_pool.is_closed
    assert calls == ["daily", "daily"]
    assert provider.call_count == 2


@pytest.mark.parametrize("status,kind", [
    (429, "transient_failure"), (500, "transient_failure"), (599, "transient_failure"),
    (401, "unauthorized"), (403, "unauthorized"), (404, "unsupported"),
    (405, "unsupported"), (410, "unsupported"), (302, "contract_failure"),
    (400, "contract_failure"), (600, "contract_failure"),
])
def test_http_status_classification_never_follows_redirects(status, kind):
    """场景：状态码覆盖与冲突消息；预期：按 HTTP 状态分类。例：401+限流正文→unauthorized。"""
    handler = Mock(return_value=httpx.Response(
        status, json={"code": 429, "msg": "访问频率超限"},
        headers={"location": "https://example.invalid"},
    ))
    with HttpTransport(lambda: PLACEHOLDER, live=True,
                       http_transport=httpx.MockTransport(handler)) as transport:
        with pytest.raises(ProviderError) as caught:
            transport.request("daily", {})
    assert caught.value.kind == kind
    assert caught.value.__context__ is None
    assert handler.call_count == 1


@pytest.mark.parametrize("payload", [
    {"code": 0, "echo": [PLACEHOLDER]}, {"code": 0, "token": "discarded"}, [1, 2],
])
def test_http_rejects_secret_echo_and_non_object_payloads(payload):
    """场景：凭据回显、敏感字段或数组正文；预期：拒绝返回。例：echo 占位凭据→失败。"""
    handler = Mock(return_value=httpx.Response(200, json=payload))
    with HttpTransport(lambda: PLACEHOLDER, live=True,
                       http_transport=httpx.MockTransport(handler)) as transport:
        with pytest.raises(ProviderError) as caught:
            transport.request("daily", {})
    assert caught.value.kind == "contract_failure"
    assert PLACEHOLDER not in str(caught.value)


@pytest.mark.parametrize("failure,kind", [
    (httpx.ConnectError(PLACEHOLDER), "transient_failure"),
    (ValueError(PLACEHOLDER), "contract_failure"),
])
def test_http_sanitizes_transport_errors(failure, kind):
    """场景：底层网络或协议异常；预期：固定消息、无异常链。例：ConnectError→临时失败。"""
    handler = Mock(side_effect=failure)
    with HttpTransport(lambda: PLACEHOLDER, live=True,
                       http_transport=httpx.MockTransport(handler)) as transport:
        with pytest.raises(ProviderError) as caught:
            transport.request("daily", {})
    assert caught.value.kind == kind
    assert caught.value.__context__ is None
    assert PLACEHOLDER not in str(caught.value)


def test_http_malformed_json_and_secret_in_public_params():
    """场景：坏 JSON 或参数夹带凭据值；预期：拒绝且参数泄漏时不发送。例：回显值→零请求。"""
    handler = Mock(return_value=httpx.Response(200, content=b"not-json"))
    with HttpTransport(lambda: PLACEHOLDER, live=True,
                       http_transport=httpx.MockTransport(handler)) as transport:
        with pytest.raises(ProviderError):
            transport.request("daily", {"query": PLACEHOLDER})
        assert handler.call_count == 0
        with pytest.raises(ProviderError):
            transport.request("daily", {})
        assert handler.call_count == 1


@pytest.mark.parametrize("api_name,params", [("../daily", {}), ("daily", {"token": PLACEHOLDER})])
def test_invalid_public_inputs_rejected_before_recording(api_name, params):
    """场景：接口名穿越或凭据参数；预期：不记录也不发送。例：token 参数→空 calls。"""
    transport = FakeTransport([response()])
    with pytest.raises(ProviderError):
        transport.request(api_name, params)
    assert transport.calls == []


@pytest.mark.parametrize("failure,kind", [
    (ProviderError("unsupported", "固定测试错误"), "unsupported"),
    (httpx.ReadTimeout(PLACEHOLDER), "transient_failure"),
    (ValueError(PLACEHOLDER), "contract_failure"),
    ({"code": 0, "token": PLACEHOLDER}, "contract_failure"),
])
def test_fake_queue_errors_are_safe(failure, kind):
    """场景：离线队列包含异常或敏感响应；预期：安全分类。例：超时→transient_failure。"""
    with pytest.raises(ProviderError) as caught:
        FakeTransport([failure]).request("daily", {})
    assert caught.value.kind == kind
    assert PLACEHOLDER not in str(caught.value)


def test_fake_queue_clones_inputs_and_detects_exhaustion():
    """场景：参数后续被调用方修改；预期：历史调用不变。例：a 从 1 改 2→历史仍为 1。"""
    transport = FakeTransport([response([[1]])])
    params = {"a": [1]}
    payload = transport.request("daily", params)
    params["a"].append(2)
    assert transport.calls == [{"api_name": "daily", "params": {"a": [1]}}]
    assert payload == response([[1]])
    with pytest.raises(ProviderError, match="耗尽"):
        transport.request("daily", {})


@pytest.mark.parametrize("kwargs", [
    {"rate_per_minute": 0}, {"rate_per_minute": float("inf")}, {"rates": {"daily": -1}},
])
def test_limiter_rejects_invalid_rates(kwargs):
    """场景：限速配置为零、负数或无穷；预期：构造拒绝。例：速率0→ValueError。"""
    with pytest.raises(ValueError):
        Limiter(**kwargs)


def test_limiter_shares_account_api_budget_across_clients(tmp_path):
    """场景：两客户端共享账户接口；预期：共享预算且其他接口独立。例：第二请求→等一秒。"""
    sleeps = []
    limiter = Limiter(120, rates={"daily": 60}, clock=lambda: 0.0, sleep=sleeps.append)
    first = client_for(tmp_path, [response()], account="research", limiter=limiter)
    second = client_for(tmp_path, [response()], account="research", limiter=limiter)
    first.fetch("daily", {}, OBSERVED)
    second.fetch("daily", {}, OBSERVED)
    limiter.acquire("research", "weekly")
    limiter.acquire("other", "daily")
    limiter.acquire("research", "daily", 500)
    assert sleeps == [1.0, 2.0]


@pytest.mark.parametrize("account,rate", [
    ("bad alias", None), ("public", 0), ("public", float("nan")),
])
def test_limiter_acquire_rejects_invalid_alias_and_override(account, rate):
    """场景：无效公开别名或速率覆盖；预期：预约前拒绝。例：带空格别名→ValueError。"""
    with pytest.raises(ValueError):
        Limiter().acquire(account, "daily", rate)


@pytest.mark.parametrize("kwargs", [
    {"account": "bad alias"}, {"retries": -1}, {"retries": True},
    {"retry_base": float("inf")}, {"rate_per_minute": 0},
])
def test_client_rejects_invalid_configuration(tmp_path, kwargs):
    """场景：无效重试、别名或速率；预期：初始化拒绝。例：retries=-1→ValueError。"""
    with pytest.raises(ValueError):
        client_for(tmp_path, [], **kwargs)


@pytest.mark.parametrize("payload,kind", [
    ({}, "contract_failure"), ({"code": False}, "contract_failure"),
    ({"code": -2001, "msg": "积分不足，权限未开通"}, "unauthorized"),
    ({"code": -1, "msg": "接口不存在"}, "unsupported"),
    ({"code": -1, "msg": "参数错误"}, "contract_failure"),
    ({"code": 401, "msg": "访问频率超限"}, "unauthorized"),
    ({"code": 403, "msg": "too many requests"}, "unauthorized"),
    ({"code": -999, "msg": "失败"}, "contract_failure"),
])
def test_provider_non_transient_failures_are_not_empty_or_retried(tmp_path, payload, kind):
    """场景：权限、参数及未知错误；预期：不重试也不产空数据。例：参数错→一次调用零文件。"""
    client = client_for(tmp_path, [payload, response()], retries=3)
    with pytest.raises(ProviderError) as caught:
        client.fetch("daily", {}, OBSERVED)
    assert caught.value.kind == kind
    assert len(client.transport.calls) == 1
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("payload", [
    {"code": 429},
    {"code": -2001, "msg": "每分钟最多访问该接口200次，详见权限说明"},
    {"code": -2001, "msg": "访问频率超限"},
])
def test_explicit_transient_provider_errors_retry_then_preserve_success(tmp_path, payload):
    """场景：明确数字或文字限流；预期：重试后只保存成功响应。例：负码限流→两次一文件。"""
    sleeps = []
    client = client_for(tmp_path, [payload, response([[1]])], sleep=sleeps.append, jitter=lambda: 0)
    page = client.fetch("daily", {}, OBSERVED)
    assert page.fields == ["id"] and page.rows == [{"id": 1}]
    assert sleeps == [0.5]
    assert len(client.transport.calls) == 2
    assert len(list(tmp_path.rglob("*.json"))) == 1


def test_custom_temporary_code_retry_exhaustion_is_bounded(tmp_path):
    """场景：已配置临时代码一直失败；预期：严格限制次数。例：retries=2→三次请求。"""
    sleeps = []
    client = client_for(tmp_path, [{"code": -700}] * 3, transient_codes=frozenset({-700}),
                        retries=2, sleep=sleeps.append, jitter=lambda: 0.25)
    with pytest.raises(ProviderError) as caught:
        client.fetch("daily", {}, OBSERVED)
    assert caught.value.kind == "transient_failure"
    assert sleeps == [0.75, 1.25]
    assert len(client.transport.calls) == 3


@pytest.mark.parametrize("failure,kind", [
    (httpx.ReadError(PLACEHOLDER), "transient_failure"),
    (RuntimeError(PLACEHOLDER), "contract_failure"),
])
def test_custom_transport_exception_is_sanitized(tmp_path, failure, kind):
    """场景：第三方传输直接抛底层错误；预期：重建安全错误。例：RuntimeError→无异常链。"""
    transport = Mock(request=Mock(side_effect=failure))
    limiter = Limiter(clock=lambda: 0.0, sleep=lambda delay: None)
    client = Client(transport, RawStore(tmp_path), limiter=limiter, retries=0)
    with pytest.raises(ProviderError) as caught:
        client.fetch("daily", {}, OBSERVED)
    assert caught.value.kind == kind
    assert caught.value.__context__ is None
    assert PLACEHOLDER not in str(caught.value)


def test_production_clients_default_to_shared_limiter(tmp_path):
    """场景：非离线传输未指定限速器；预期：复用默认预算。例：两个 Client→同一 Limiter。"""
    transport = Mock(request=Mock(return_value=response()))
    first = Client(transport, RawStore(tmp_path))
    second = Client(transport, RawStore(tmp_path))
    assert first.limiter is second.limiter
    assert isinstance(first.limiter, Limiter)


@pytest.mark.parametrize("jitter", [float("nan"), -1])
def test_retry_invalid_jitter_fails_closed(tmp_path, jitter):
    """场景：注入抖动返回非法值；预期：不等待也不续取。例：负抖动→contract_failure。"""
    client = client_for(tmp_path, [{"code": 429}], jitter=lambda: jitter)
    with pytest.raises(ProviderError, match="抖动"):
        client.fetch("daily", {}, OBSERVED)


@pytest.mark.parametrize("failure", [OSError("磁盘故障"), ValueError("校验失败")])
def test_fetch_requires_raw_store_success_before_parsing(tmp_path, failure):
    """场景：原始保存失败；预期：不进入解析。例：磁盘错误→contract_failure。"""
    client = client_for(tmp_path, [response([[1]])])
    client.raw_store = Mock(write=Mock(side_effect=failure))
    client._parse = Mock()
    with pytest.raises(ProviderError, match="保存或校验"):
        client.fetch("daily", {}, OBSERVED)
    client._parse.assert_not_called()


@pytest.mark.parametrize("data", [
    None, {"fields": [], "items": []}, {"fields": ["id", "id"], "items": []},
    {"fields": [None], "items": []}, {"fields": ["id"], "items": {}},
    {"fields": ["id"], "items": [[1, 2]]}, {"fields": ["id"], "items": [1]},
])
def test_bad_page_schema_is_archived_before_rejection(tmp_path, data):
    """场景：成功状态但数据结构错误；预期：留原始证据后失败。例：行宽错误→一原始文件。"""
    client = client_for(tmp_path, [{"code": 0, "data": data}])
    with pytest.raises(ProviderError) as caught:
        client.fetch("daily", {}, OBSERVED)
    assert caught.value.kind == "contract_failure"
    assert len(list(tmp_path.rglob("*.json"))) == 1


def test_parse_rechecks_sensitive_fields_after_raw_boundary(tmp_path):
    """场景：外部构造的 RawRecord 含敏感列；预期：解析层再次拒绝。例：token 列→失败。"""
    client = client_for(tmp_path, [])
    record = RawRecord("public", "public", OBSERVED, Path("unused"),
                       response([[PLACEHOLDER]], ["token"]))
    with pytest.raises(ProviderError, match="敏感"):
        client._parse(record)


def test_pagination_requires_short_terminal_page_and_preserves_params(tmp_path):
    """场景：整页后接短页；预期：偏移正确、原参数不变。例：两行加一行→两页。"""
    client = client_for(tmp_path, [response([[1], [2]]), response([[3]])])
    params = {"symbol": "PUBLIC"}
    pages = client.pages("daily", params, OBSERVED, ["id"], 2, 3)
    assert [page.rows for page in pages] == [[{"id": 1}, {"id": 2}], [{"id": 3}]]
    assert [call["params"]["offset"] for call in client.transport.calls] == [0, 2]
    assert all(call["params"]["limit"] == 2 for call in client.transport.calls)
    assert params == {"symbol": "PUBLIC"}


def test_non_paginated_short_result_does_not_inject_offsets(tmp_path):
    """场景：不可分页接口返回短结果；预期：不添加分页参数。例：一行、上限二→一页。"""
    client = client_for(tmp_path, [response([[1]])])
    pages = client.pages("daily", {}, OBSERVED, ["id"], 2, 3, paginate=False)
    assert len(pages) == 1
    assert client.transport.calls == [{"api_name": "daily", "params": {}}]


@pytest.mark.parametrize("params,key,size,maximum", [
    ({}, ["id"], 0, 1), ({}, ["id"], 1, 0), ({}, [], 1, 1),
    ({}, ["id", "id"], 1, 1), ({}, ["id"], True, 1),
    ({"offset": -1}, ["id"], 1, 1), ({"offset": "0"}, ["id"], 1, 1),
    ({"limit": 1}, ["id"], 2, 1), ({"limit": True}, ["id"], 1, 1),
])
def test_pagination_invalid_contract_is_rejected_before_requests(
    tmp_path, params, key, size, maximum,
):
    """场景：分页或主键配置无效；预期：零次请求。例：limit 与页大小不同→拒绝。"""
    client = client_for(tmp_path, [])
    with pytest.raises(ProviderError):
        client.pages("daily", params, OBSERVED, key, size, maximum)
    assert client.transport.calls == []


@pytest.mark.parametrize("payloads,key,size,maximum,paginate,fragment", [
    ([response([[1]]), response([[2]], ["other"])], ["id"], 1, 3, True, "字段发生变化"),
    ([response([[1]], ["other"])], ["id"], 2, 3, True, "缺少完整主键"),
    ([response([[1], [2]])], ["id"], 1, 3, True, "超过声明页上限"),
    ([response([[None]])], ["id"], 2, 3, True, "主键为空"),
    ([response([[""]])], ["id"], 2, 3, True, "主键为空"),
    ([response([[1]]), response([[1]])], ["id"], 1, 3, True, "主键重复"),
    ([response([[1, "a"]], ["id", "value"]), response([[1, "b"]], ["id", "value"])],
     ["id"], 1, 3, True, "主键冲突"),
    ([response([[1]])], ["id"], 1, 1, True, "最大页数"),
    ([response([[1]])], ["id"], 1, 3, False, "非分页接口"),
])
def test_pagination_incomplete_or_inconsistent_data_fails_closed(
    tmp_path, payloads, key, size, maximum, paginate, fragment,
):
    """场景：重复、冲突、字段漂移或截断；预期：禁止返回不完整页集。例：满上限→要求分片。"""
    client = client_for(tmp_path, payloads)
    with pytest.raises(ProviderError, match=fragment) as caught:
        client.pages("daily", {}, OBSERVED, key, size, maximum, paginate=paginate)
    assert caught.value.kind == "contract_failure"
