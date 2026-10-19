"""研究证券范围与产品主板规则一致，拒绝相似代码。"""

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.experiment_scope import (
    account_scope_mask,
    mainboard_scope_mask,
)


def test_mainboard_scope_includes_both_exchanges_and_excludes_other_boards():
    """场景：主板、创业板、科创板、B股及北交所混合；预期仅沪深主板为真。"""
    codes = [
        "600000.SH", "601000.SH", "603000.SH", "605000.SH",
        "000001.SZ", "001001.SZ", "002001.SZ", "003001.SZ",
        "300001.SZ", "301001.SZ", "688001.SH", "689001.SH",
        "900001.SH", "200001.SZ", "920001.BJ", "600000.SZ", "600000.SH.extra",
    ]
    assert mainboard_scope_mask(codes).tolist() == [True] * 8 + [False] * 9


def test_empty_scope_preserves_boolean_axis():
    """场景：空证券轴；预期空布尔数组，不出现浮点掩码。"""
    result = mainboard_scope_mask([])
    assert result.shape == (0,)
    assert result.dtype == bool


def test_original_scope_preserves_historical_eligibility_without_new_admissions():
    """原范围不新增板块限制；已有不合资格证券仍不能因掩码为真而入选。"""
    codes = ["600000.SH", "300001.SZ", "688001.SH", "920001.BJ"]
    eligible = np.asarray([[True, True, False, False]])
    original = eligible & account_scope_mask(codes, "original")[None, :]
    assert np.array_equal(original, eligible)
    mainboard = eligible & account_scope_mask(codes, "mainboard")[None, :]
    assert mainboard.tolist() == [[True, False, False, False]]


def test_original_scope_empty_axis_and_unknown_scope():
    """空原范围保持布尔轴；拼错范围必须失败，不静默放开筛选。"""
    original = account_scope_mask([], "original")
    assert original.shape == (0,)
    assert original.dtype == bool
    with pytest.raises(ValueError, match="unknown_account_scope"):
        account_scope_mask(["600000.SH"], "typo")
