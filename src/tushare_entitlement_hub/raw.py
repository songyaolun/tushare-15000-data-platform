"""在规范化前保存可校验的原始响应，禁止凭据进入持久化边界。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SENSITIVE_NAMES = frozenset({
    "token", "accesstoken", "refreshtoken", "idtoken", "tusharetoken", "apitoken",
    "authorization", "proxyauthorization", "password", "passwd", "secret", "clientsecret",
    "apikey", "privatekey", "cookie", "setcookie", "credentials", "credential", "certificate",
})
_ASSIGNMENT = re.compile(
    r"(?i)(?:token|authorization|password|api[_-]?key|secret|cookie)\s*[:=]\s*\S+"
)


def _possible_assignment(value: str) -> bool:
    """只筛掉绝不匹配赋值正则的文本，不代替最终安全检查；例：普通大段JSON返回False。

    使用内建casefold避免字符串子类覆写；补齐Python Unicode IGNORECASE的点上/无点i，
    长s和Kelvin符号已由casefold处理。允许筛选假阳性，最终仍对原文本运行完整正则。
    例：'API_KEY=<REDACTED>'和'authorİzation:<REDACTED>'都返回True，不能绕过检查。
    """
    folded = str.casefold(value).replace("i\u0307", "i").replace("\u0131", "i")
    return any(
        name in folded
        for name in (
            "token",
            "authorization",
            "password",
            "api_key",
            "api-key",
            "apikey",
            "secret",
            "cookie",
        )
    )


def assert_public(value: Any) -> None:
    """递归拒绝凭据字段与文本形式的凭据赋值，错误中不带输入内容。

    参数为准备落盘的 JSON 数据；正常时返回 None，发现敏感字段时抛 ValueError。
    示例：assert_public({'ts_code': '000001.SZ'}) 返回 None；
    assert_public({'nested': {'token': '<REDACTED>'}}) 抛出 ValueError。
    实际凭据值的回显另由发送时持有凭据的传输层检查，本函数不会读取任何环境变量。
    赋值匹配必需分隔符及关键词；缺少必要条件时跳过不可能命中的正则，不缓存任何值。
    """
    if isinstance(value, dict):
        for name, child in value.items():
            if not isinstance(name, str):
                raise ValueError("JSON 对象字段名必须为字符串")
            normalized = re.sub(r"[^a-z0-9]", "", name.lower())
            if normalized in _SENSITIVE_NAMES or normalized.endswith(("token", "secret")):
                raise ValueError("拒绝保存含凭据字段的数据")
            if name == "fields" and isinstance(child, list):
                for field in child:
                    if isinstance(field, str):
                        assert_public({field: None})
            assert_public(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_public(child)
    elif (
        isinstance(value, str)
        and (str.__contains__(value, ":") or str.__contains__(value, "="))
        and _possible_assignment(value)
        and _ASSIGNMENT.search(value)
    ):
        raise ValueError("拒绝保存疑似凭据赋值的数据")


def canonical_json(value: Any) -> str:
    """将公开 JSON 数据序列化为稳定文本，拒绝 NaN 和不可序列化的对象。

    参数为 JSON 兼容对象，返回按字段名排序、没有多余空格的 Unicode 字符串。
    示例：canonical_json({'b': 2, 'a': 1}) 返回 '{"a":1,"b":2}'。
    所有失败均使用固定错误文本，避免异常信息携带输入值。
    """
    assert_public(value)
    result: str | None = None
    try:
        result = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                            allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        pass
    if result is None:
        raise ValueError("数据必须是有限数值组成的 JSON 对象")
    return result


@dataclass(frozen=True)
class RawRecord:
    """一次经读回校验的观测，response 保留提供方原始字段和单位。"""

    response_hash: str
    request_id: str
    observed_at: str
    path: Path
    response: dict[str, Any]


class RawStore:
    """基于内容哈希路径保存响应；相同观测可幂等重放，不覆盖其他观测。"""

    def __init__(self, root: Path) -> None:
        """设置原始响应根目录，构造时不创建文件。

        示例：RawStore(Path('/tmp/raw')).root 返回 Path('/tmp/raw')。
        """
        self.root = Path(root)

    def write(
        self, api_name: str, params: dict[str, Any], response: dict[str, Any], observed_at: str
    ) -> RawRecord:
        """原子保存成功响应，读回并验证完整哈希后返回 RawRecord。

        api_name 是接口名，params 是无凭据请求参数，observed_at 是显式抓取时间。
        request_id 只取接口和参数的稳定哈希；响应哈希不包含凭据或账户信息。
        示例：write('daily', {}, {'code': 0, 'data': {}}, '2026-01-01T00:00:00Z')
        返回包含 JSON 文件路径、响应哈希及原始 data 字段的 RawRecord。
        不同抓取时间分别保留，完全相同的观测重复写入使用同一文件。
        """
        if type(response.get("code")) is not int or response["code"] != 0:
            raise ValueError("只有成功响应可以进入原始数据层")
        return self._write(api_name, params, response, observed_at)

    def write_error(
        self, api_name: str, params: dict[str, Any], response: dict[str, Any], observed_at: str
    ) -> RawRecord:
        """Preserve a provider rejection separately from successful raw pages.

        Example: code=50101 produces errors/<api>/... with the original error code;
        code=0 is rejected here. Public-data checks and atomic readback still apply.
        """
        if type(response.get("code")) is not int or response["code"] == 0:
            raise ValueError("错误证据必须包含非零提供方状态码")
        return RawStore(self.root / "errors")._write(
            api_name, params, response, observed_at, error=True
        )

    def _write(
        self, api_name: str, params: dict[str, Any], response: dict[str, Any], observed_at: str,
        *, error: bool = False,
    ) -> RawRecord:
        """Atomically preserve public bytes and verify their complete envelope hash.

        Example: an error envelope includes evidence_kind=provider_error_not_successful_raw.
        """
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", api_name):
            raise ValueError("接口名格式无效")
        if not isinstance(observed_at, str) or not observed_at.strip():
            raise ValueError("必须提供抓取时间")
        response_text = canonical_json(response)
        request_text = canonical_json({"api_name": api_name, "params": params})
        response_hash = hashlib.sha256(response_text.encode()).hexdigest()
        request_id = hashlib.sha256(request_text.encode()).hexdigest()
        envelope = {
            "api_name": api_name, "params": params, "observed_at": observed_at,
            "request_id": request_id, "response_hash": response_hash, "response": response,
        }
        if error:
            envelope["evidence_kind"] = "provider_error_not_successful_raw"
        encoded = canonical_json(envelope).encode()
        content_hash = hashlib.sha256(encoded).hexdigest()
        directory = self.root / api_name / request_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{content_hash}.json"
        temporary: Path | None = None
        try:
            if not path.exists():
                with tempfile.NamedTemporaryFile(
                    dir=directory, suffix=".tmp", delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            stored = path.read_bytes()
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        if stored != encoded or hashlib.sha256(stored).hexdigest() != content_hash:
            raise ValueError("原始响应读回校验失败")
        decoded: dict[str, Any] = json.loads(stored)
        stored_response: dict[str, Any] = decoded["response"]
        if hashlib.sha256(canonical_json(stored_response).encode()).hexdigest() != response_hash:
            raise ValueError("原始响应内容哈希不一致")
        return RawRecord(response_hash, request_id, observed_at, path, stored_response)
