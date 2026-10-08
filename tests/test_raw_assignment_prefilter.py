"""凭据正则的Unicode等价必要条件，长JSON只减少无效扫描，不降低公开边界。"""

import itertools
import re
from unittest.mock import Mock

import pytest

from tushare_entitlement_hub import raw


def test_every_credential_case_variant_still_reaches_original_regex():
    """ASCII及IGNORECASE特殊Unicode逐位置变体均不能被预筛选漏掉，非赋值文本保持可用。"""
    matched = 0
    for word in (
        "token",
        "authorization",
        "password",
        "api_key",
        "api-key",
        "apikey",
        "secret",
        "cookie",
    ):
        variants = [
            tuple(
                dict.fromkeys(
                    (letter, letter.upper(), *({"i": "İı", "s": "ſ", "k": "K"}.get(letter, "")))
                )
            )
            for letter in word
        ]
        for letters in itertools.product(*variants):
            text = "".join(letters) + " = <REDACTED>"
            if raw._ASSIGNMENT.search(text):
                matched += 1
                assert raw._possible_assignment(text)
                with pytest.raises(ValueError, match="疑似凭据赋值"):
                    raw.assert_public(text)
    assert matched > 10000
    raw.assert_public("普通token文字，无赋值")
    raw.assert_public("业务字段: a; apikey只是名字")


def test_long_json_without_keywords_avoids_regex_and_string_subclass_cannot_bypass(monkeypatch):
    """长JSON没有任何凭据关键字时不调用正则；子类伪造casefold仍拒绝原文本中的赋值。"""
    original = raw._ASSIGNMENT
    matcher = Mock(wraps=original)
    monkeypatch.setattr(raw, "_ASSIGNMENT", matcher)
    raw.assert_public('{"synthetic":"' + "a" * 100000 + '"}')
    assert matcher.search.call_count == 0

    class ForgedText(str):
        """伪造大小写折叠方法，预期安全边界仅使用内建str实现。"""

        def casefold(self):
            """返回假的无凭据文本；例：安全实现不得调用此覆写。"""
            return "ordinary"

    with pytest.raises(ValueError):
        raw.assert_public(ForgedText("api_Key=<REDACTED>"))
    assert matcher.search.call_count == 1
    # casefold可产生假阳性，但原正则仍保留最终判定权，不扩大拒绝集合。
    text = "paßword=<REDACTED>"
    assert raw._possible_assignment(text)
    assert re.search(original, text) is None
    raw.assert_public(text)
