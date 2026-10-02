"""只在发送请求前读取用户已有凭据，不复制、打印或持久化凭据。"""

from __future__ import annotations

import os
from pathlib import Path


def configured_credential(path: Path | None = None) -> str | None:
    """读取显式 dotenv 路径或 TUSHARE_TOKEN 环境变量，不执行配置文件内容。

    输入可选用户指定路径，返回凭据或 None；文件错误用固定文本，不回显任何内容。
    示例：环境未配置且 path=None 返回 None；显式文件优先于环境变量。
    """
    if path is None:
        return os.environ.get("TUSHARE_TOKEN") or None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.strip().removeprefix("export ").partition("=")
            if separator and name.strip() == "TUSHARE_TOKEN":
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                    value = value[1:-1]
                if value:
                    return value
    except (OSError, UnicodeError):
        pass
    raise ValueError("credential_file_unavailable_or_missing_value")
