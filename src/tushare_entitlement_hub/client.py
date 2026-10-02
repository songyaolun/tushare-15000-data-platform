"""离线优先的 Tushare Pro 传输、共享限速、有限重试与严格分页。"""

from __future__ import annotations

import json
import math
import random
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx

from tushare_entitlement_hub.raw import RawRecord, RawStore, assert_public, canonical_json

ErrorKind = Literal["unauthorized", "unsupported", "transient_failure", "contract_failure"]


class ProviderError(Exception):
    """只有固定安全消息的提供方错误，kind 可直接用于探测结果分类。"""

    def __init__(self, kind: ErrorKind, message: str) -> None:
        """保存错误类别及调用方提供的安全说明，不保存响应和请求对象。

        示例：ProviderError('unauthorized', '访问未获授权').kind 返回 'unauthorized'。
        调用方须使用固定消息，不能把提供方原文或凭据交给 message。
        """
        self.kind = kind
        super().__init__(message)


class Transport(Protocol):
    """公开参数进、公开 JSON 出的传输边界，凭据不进入客户端或原始层。"""

    def request(self, api_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """请求一个接口并返回原始响应，失败时抛安全的 ProviderError。

        示例：request('daily', {}) 返回 {'code': 0, 'data': {'fields': [], 'items': []}}。
        """
        ...


def _safe_inputs(api_name: str, params: dict[str, Any]) -> None:
    """发送前校验接口名称与公开 JSON 参数，转换为固定契约错误。

    示例：_safe_inputs('daily', {}) 返回 None；接口名 '../daily' 会抛 ProviderError。
    """
    valid = bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", api_name))
    try:
        canonical_json(params)
    except ValueError:
        valid = False
    if not valid:
        raise ProviderError("contract_failure", "接口名称或公开参数无效")


def _contains_secret(value: Any, secret: str) -> bool:
    """在内存中检查当前凭据的原文回显，不保存、不打印也不哈希该凭据。

    示例：_contains_secret({'msg': '<REDACTED>'}, '<REDACTED>') 返回 True。
    参数 secret 仅由发送函数短暂传入，返回是否在字段名或字符串值中出现。
    """
    if isinstance(value, str):
        return secret in value
    if isinstance(value, dict):
        return any(_contains_secret(key, secret) or _contains_secret(child, secret)
                   for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_secret(child, secret) for child in value)
    return False


class HttpTransport:
    """连接复用的 HTTPS POST 传输；默认关闭真实调用且不读取环境变量。"""

    def __init__(
        self, token_provider: Callable[[], str], *, live: bool = False,
        base_url: str = "https://api.tushare.pro", timeout: float = 30.0,
        http_transport: httpx.BaseTransport | None = None,
    ) -> None:
        """创建禁重定向、禁环境代理的共享连接池。

        token_provider 只在 request 即将发送时调用；live 必须显式为 True。
        http_transport 可传 httpx.MockTransport，以离线检查实际 HTTP 请求。
        示例：HttpTransport(lambda: '<REDACTED>', http_transport=mock).request('daily', {})
        因默认 live=False 抛 unauthorized，且不会调用 token_provider。
        目标只允许官方 HTTPS 主机根路径，拒绝自定义主机、端口、查询和用户信息。
        """
        if base_url not in {"https://api.tushare.pro", "https://api.tushare.pro/"}:
            raise ProviderError("contract_failure", "只允许官方 HTTPS API 地址")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ProviderError("contract_failure", "请求超时必须为正数")
        self._token_provider = token_provider
        self._live = live
        self._base_url = base_url
        self._client = httpx.Client(
            timeout=timeout, follow_redirects=False, trust_env=False, transport=http_transport,
        )

    def request(self, api_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """发送公开参数，在离开函数前拒绝所有凭据回显并丢弃原始异常链。

        示例：在显式 live=True 且注入离线 MockTransport 时，request('daily', {})
        返回 mock 提供的成功 JSON；HTTP 429 则抛 kind='transient_failure'。
        不返回 httpx.Response，避免其 request 属性把凭据带给上层。
        """
        _safe_inputs(api_name, params)
        if not self._live:
            raise ProviderError("unauthorized", "真实接口调用需要显式启用")
        token: str | None = None
        try:
            token = self._token_provider()
        except Exception:
            pass
        if not isinstance(token, str) or not token.strip():
            raise ProviderError("unauthorized", "未取得有效访问凭据")
        if _contains_secret(params, token) or token in api_name:
            token = None
            raise ProviderError("contract_failure", "公开请求参数包含访问凭据")
        payload: Any = None
        error: tuple[ErrorKind, str] | None = None
        try:
            response = self._client.post(self._base_url, json={
                "api_name": api_name, "token": token, "params": params, "fields": "",
            })
            if response.status_code == 429 or 500 <= response.status_code <= 599:
                error = ("transient_failure", "提供方暂时不可用")
            elif response.status_code in {401, 403}:
                error = ("unauthorized", "提供方拒绝访问")
            elif response.status_code in {404, 405, 410}:
                error = ("unsupported", "提供方不支持此请求")
            elif response.status_code != 200:
                error = ("contract_failure", "提供方返回非预期 HTTP 状态")
            else:
                payload = response.json()
                if _contains_secret(payload, token):
                    payload = None
                    error = ("contract_failure", "响应包含凭据回显，已拒绝保存")
                else:
                    assert_public(payload)
        except httpx.RequestError:
            error = ("transient_failure", "网络请求暂时失败")
        except Exception:
            error = ("contract_failure", "传输或响应格式不符合契约")
        finally:
            self._client.cookies.clear()
            token = None
        if error is not None:
            raise ProviderError(*error)
        if not isinstance(payload, dict):
            raise ProviderError("contract_failure", "提供方响应必须是 JSON 对象")
        return payload

    def close(self) -> None:
        """关闭复用连接池，释放连接资源。

        示例：transport.close() 返回 None；之后不应继续发送请求。
        """
        self._client.close()

    def __enter__(self) -> HttpTransport:
        """进入上下文并返回当前传输对象。

        示例：with HttpTransport(provider) as transport 中 transport 即当前实例。
        """
        return self

    def __exit__(self, *arguments: Any) -> None:
        """离开上下文时关闭连接，不吞掉调用方异常。

        示例：退出 with 块后返回 None，块内异常继续向上传递。
        """
        self.close()


class FakeTransport:
    """按顺序消耗固定响应或安全错误，只记录公开调用参数。"""

    def __init__(self, responses: Sequence[dict[str, Any] | Exception]) -> None:
        """复制离线响应队列，不做任何真实网络调用。

        示例：FakeTransport([{'code': 0, 'data': {}}]).calls 初始为 []。
        """
        self._responses = deque(responses)
        self.calls: list[dict[str, Any]] = []

    def request(self, api_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """记录接口及参数并返回下一响应，队列耗尽即报告契约失败。

        示例：FakeTransport([{'code': 0}]).request('daily', {}) 返回 {'code': 0}。
        注入 httpx.RequestError 模拟可重试网络失败，其他异常统一清理错误文本。
        """
        _safe_inputs(api_name, params)
        self.calls.append({"api_name": api_name, "params": json.loads(canonical_json(params))})
        if not self._responses:
            raise ProviderError("contract_failure", "离线响应队列已耗尽")
        result = self._responses.popleft()
        if isinstance(result, ProviderError):
            raise ProviderError(result.kind, "离线队列模拟提供方失败")
        if isinstance(result, httpx.RequestError):
            raise ProviderError("transient_failure", "离线队列模拟网络失败")
        if isinstance(result, Exception):
            raise ProviderError("contract_failure", "离线队列模拟契约失败")
        safe: dict[str, Any] | None = None
        try:
            safe = json.loads(canonical_json(result))
        except ValueError:
            pass
        if safe is None:
            raise ProviderError("contract_failure", "离线响应包含非公开数据")
        return safe


class Limiter:
    """按账户别名及接口预约请求时间，多客户端可共享同一线程安全实例。"""

    def __init__(
        self, rate_per_minute: float = 60, *, rates: dict[str, float] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """配置默认和逐接口速率，注入时钟与 sleep 可做确定性离线验证。

        示例：Limiter(120, rates={'daily': 60}) 对 daily 每秒最多预约一次。
        输入速率必须为正数；实例只保存公开账户别名，不接收访问凭据。
        """
        configured = dict(rates or {})
        if (not math.isfinite(rate_per_minute) or rate_per_minute <= 0
                or any(not math.isfinite(rate) or rate <= 0 for rate in configured.values())):
            raise ValueError("每分钟调用次数必须为正数")
        self.rate_per_minute = rate_per_minute
        self.rates = configured
        self._clock = clock
        self._sleep = sleep
        self._next: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()

    def acquire(self, account: str, api_name: str, rate_per_minute: float | None = None) -> None:
        """为账户和接口预约一个时间槽，等待完成后返回 None。

        示例：同一账户连续 acquire('research', 'daily') 两次且速率为 60，
        第二次等待约一秒；另一个接口或账户使用独立时间槽。
        显式速率比实例配置更严格时生效，不能通过客户端覆盖放宽共享限速。
        """
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", account):
            raise ValueError("账户必须使用简短公开别名")
        configured = self.rates.get(api_name, self.rate_per_minute)
        if rate_per_minute is not None and (
            not math.isfinite(rate_per_minute) or rate_per_minute <= 0
        ):
            raise ValueError("每分钟调用次数必须为正数")
        rate = configured if rate_per_minute is None else min(configured, rate_per_minute)
        key = (account, api_name)
        with self._lock:
            now = self._clock()
            scheduled = max(now, self._next.get(key, now))
            self._next[key] = scheduled + 60.0 / rate
        delay = scheduled - now
        if delay > 0:
            self._sleep(delay)


_SHARED_LIMITER = Limiter()


@dataclass(frozen=True)
class Page:
    """经过原始落盘和结构验证的单页数据，不表示整个分片采集完成。"""

    fields: list[str]
    rows: list[dict[str, Any]]
    raw: RawRecord


class Client:
    """组合传输、有限重试与分页契约；任何失败都不会转换为成功空页。"""

    def __init__(
        self, transport: Transport, raw_store: RawStore, account: str = "offline",
        limiter: Limiter | None = None, retries: int = 2, *, rate_per_minute: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random, retry_base: float = 0.5,
        transient_codes: frozenset[int] = frozenset({429, 500, 502, 503, 504}),
    ) -> None:
        """注入可替换组件；retries 表示首次请求之外允许的重试次数。

        示例：Client(FakeTransport([response]), RawStore(path), retries=0) 不做重试。
        生产实例默认共享限速器，离线 FakeTransport 默认省略真实等待；显式传入
        limiter 时仍执行其限速。transient_codes 只应配置已确认可重试的提供方代码。
        """
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", account):
            raise ValueError("账户必须使用简短公开别名")
        if (type(retries) is not int or not 0 <= retries <= 20
                or not math.isfinite(retry_base) or retry_base < 0
                or (rate_per_minute is not None and (
                    not math.isfinite(rate_per_minute) or rate_per_minute <= 0))):
            raise ValueError("重试及限速参数无效")
        self.transport = transport
        self.raw_store = raw_store
        self.account = account
        self.limiter = limiter if limiter is not None else (
            None if isinstance(transport, FakeTransport) else _SHARED_LIMITER
        )
        self.retries = retries
        self.rate_per_minute = rate_per_minute
        self._sleep = sleep
        self._jitter = jitter
        self._retry_base = retry_base
        self._transient_codes = transient_codes

    def _check_status(self, response: dict[str, Any]) -> None:
        """分类提供方状态且不传播其消息正文，成功时返回 None。

        示例：{'code': 0} 通过；{'code': 429} 抛 transient_failure。
        参数错误与未知错误均关闭流程，不依据模糊消息把未知错误当作临时失败。
        """
        code = response.get("code")
        if type(code) is not int:
            raise ProviderError("contract_failure", "提供方状态码缺失或格式无效")
        if code == 0:
            return
        if code in {401, 403}:
            raise ProviderError("unauthorized", "接口访问未获授权")
        message = str(response.get("msg", "")).lower()
        if any(phrase in message for phrase in (
            "每分钟最多访问", "访问频率超限", "rate limit exceeded", "too many requests",
        )):
            raise ProviderError("transient_failure", "提供方明确报告访问频率超限")
        if any(word in message for word in ("权限", "积分", "token", "permission", "unauthorized")):
            raise ProviderError("unauthorized", "接口访问权限不足或凭据无效")
        if any(word in message for word in (
            "接口不存在", "请指定正确的接口名", "不支持",
            "deprecated", "unsupported", "api not found",
        )):
            raise ProviderError("unsupported", "接口不可用或不受支持")
        if any(word in message for word in ("参数", "parameter", "argument")):
            raise ProviderError("contract_failure", "提供方拒绝请求参数")
        if code in self._transient_codes:
            raise ProviderError("transient_failure", "提供方明确报告临时故障")
        raise ProviderError("contract_failure", "提供方拒绝参数或返回未知错误")

    def fetch(self, api_name: str, params: dict[str, Any], observed_at: str) -> Page:
        """抓取一页，仅对临时失败重试；成功原文落盘完成后才解析行。

        示例：提供方字段 ['ts_code'] 和行 [['000001.SZ']] 会转换成
        Page.rows == [{'ts_code': '000001.SZ'}]，Page.raw 指向校验过的原始文件。
        observed_at 由调用方提供，不自动替换为系统时间或业务生效时间。
        """
        _safe_inputs(api_name, params)
        if api_name == "pro_bar":
            from .bars import compose_bars

            return compose_bars(self, params, observed_at)
        response: dict[str, Any] = {}
        for attempt in range(self.retries + 1):
            failure: tuple[ErrorKind, str] | None = None
            try:
                if self.limiter is not None:
                    self.limiter.acquire(self.account, api_name, self.rate_per_minute)
                response = self.transport.request(api_name, params)
                self._check_status(response)
            except ProviderError as error:
                failure = (error.kind, "请求失败，已终止当前采集")
            except httpx.RequestError:
                failure = ("transient_failure", "网络请求暂时失败")
            except Exception:
                failure = ("contract_failure", "请求不符合传输契约")
            if failure is None:
                break
            if failure[0] != "transient_failure" or attempt == self.retries:
                raise ProviderError(*failure)
            jitter = self._jitter()
            if not math.isfinite(jitter) or jitter < 0:
                raise ProviderError("contract_failure", "重试抖动必须为有限非负数")
            self._sleep(self._retry_base * (2 ** attempt) + jitter)
        raw: RawRecord | None = None
        try:
            raw = self.raw_store.write(api_name, params, response, observed_at)
        except (ValueError, OSError):
            pass
        if raw is None:
            raise ProviderError("contract_failure", "原始响应保存或校验失败")
        return self._parse(raw)

    def _parse(self, raw: RawRecord) -> Page:
        """把已保存响应中的字段和行转成字典，结构异常即中止。

        示例：fields=['close']、items=[[1.2]] 得到 rows=[{'close': 1.2}]。
        重复字段、字段与值数量不匹配，以及敏感字段均不能进入规范化层。
        """
        data = raw.response.get("data")
        if not isinstance(data, dict):
            raise ProviderError("contract_failure", "响应缺少数据对象")
        fields = data.get("fields")
        items = data.get("items")
        if (not isinstance(fields, list) or not fields
                or not all(isinstance(name, str) and name for name in fields)
                or len(set(fields)) != len(fields) or not isinstance(items, list)):
            raise ProviderError("contract_failure", "响应字段或数据行格式无效")
        rows: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, list) or len(item) != len(fields):
                raise ProviderError("contract_failure", "响应字段数量与行宽不一致")
            row = dict(zip(fields, item, strict=True))
            valid = True
            try:
                assert_public(row)
            except ValueError:
                valid = False
            if not valid:
                raise ProviderError("contract_failure", "响应包含敏感数据字段")
            rows.append(row)
        return Page(fields=list(fields), rows=rows, raw=raw)

    def pages(
        self, api_name: str, params: dict[str, Any], observed_at: str,
        primary_key: list[str], page_size: int, max_pages: int, paginate: bool = True,
    ) -> list[Page]:
        """获取完整分片，只有短页证明结束时才返回全部页。

        示例：page_size=2，依次取得两行及一行时返回两页；取得两行且
        max_pages=1 时抛契约错误，调用方应缩小日期或实体分片重新采集。
        主键重复或冲突、跨页字段变化、超限及缺少主键全部关闭流程。
        paginate=False 时不注入 limit/offset，但达到页上限仍要求进一步分片。
        """
        if (type(page_size) is not int or type(max_pages) is not int
                or page_size <= 0 or max_pages <= 0 or not primary_key
                or len(set(primary_key)) != len(primary_key)):
            raise ProviderError("contract_failure", "分页边界及主键配置无效")
        offset = params.get("offset", 0)
        if type(offset) is not int or offset < 0:
            raise ProviderError("contract_failure", "分页起点无效")
        if "limit" in params and (type(params["limit"]) is not int or params["limit"] != page_size):
            raise ProviderError("contract_failure", "请求行数限制必须与声明页上限一致")
        pages: list[Page] = []
        seen: dict[str, str] = {}
        expected_fields: list[str] | None = None
        for _ in range(max_pages if paginate else 1):
            request_params = dict(params)
            if paginate:
                request_params.update(limit=page_size, offset=offset)
            page = self.fetch(api_name, request_params, observed_at)
            if expected_fields is None:
                expected_fields = page.fields
            if page.fields != expected_fields:
                raise ProviderError("contract_failure", "分页过程中响应字段发生变化")
            if not set(primary_key).issubset(page.fields):
                raise ProviderError("contract_failure", "响应缺少完整主键字段")
            if len(page.rows) > page_size:
                raise ProviderError("contract_failure", "响应行数超过声明页上限")
            for row in page.rows:
                if any(row[name] is None or row[name] == "" for name in primary_key):
                    raise ProviderError("contract_failure", "响应主键为空")
                key = canonical_json([row[name] for name in primary_key])
                content = canonical_json(row)
                if key in seen:
                    description = (
                        "分页主键重复，无法确认进展" if seen[key] == content else "分页主键冲突"
                    )
                    raise ProviderError("contract_failure", description)
                seen[key] = content
            pages.append(page)
            if len(page.rows) < page_size:
                return pages
            if not paginate:
                raise ProviderError("contract_failure", "非分页接口达到行数上限，必须进一步分片")
            offset += page_size
        raise ProviderError("contract_failure", "分页达到最大页数，必须进一步分片")
