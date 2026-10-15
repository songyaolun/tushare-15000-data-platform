"""逐产品范围证书和发布状态测试。"""

import pytest

from tushare_entitlement_hub.product_readiness import (
    ProductReadiness,
    ScopeRequirement,
    assess_product_readiness,
)


def certificate(api, dates=(), field="trade_date", status="certified"):
    """构造最小范围证书；输入接口和日期，输出 readiness 可消费字典。"""
    return {
        "api_name": api,
        "status": status,
        "parameter_dates": {field: list(dates)} if dates else {},
    }


def test_readiness_requires_certified_day_count_and_target_date():
    """场景：静态目录已认证、日表仅两日；预期：两日门槛通过，目标日缺失则降级。"""
    days = ("2026-10-08", "2026-10-09")
    requirements = (ScopeRequirement("ths_index"), ScopeRequirement("daily", 2))
    certificates = (certificate("ths_index"), certificate("daily", days))
    ready = assess_product_readiness("P02", certificates, requirements, target_date=days[-1])
    assert ready.ready and ready.certified_day_counts == {"ths_index": 0, "daily": 2}
    missing = assess_product_readiness(
        "P02", certificates, requirements, target_date="2026-10-10"
    )
    assert missing.status == "incomplete"
    assert missing.reason_codes == ("target_date_not_certified:daily",)


def test_readiness_reports_missing_insufficient_and_blocked_scopes():
    """场景：接口证书缺失、日期不足并有权限阻断；预期：blocked 且缺口全部保留。"""
    result = assess_product_readiness(
        "P02",
        (certificate("daily", ("2026-10-09",)),),
        (ScopeRequirement("daily", 2), ScopeRequirement("ths_hot", 1)),
        target_date="2026-10-09",
        blocked_reasons=("unauthorized:ths_hot",),
    )
    assert result.status == "blocked"
    assert set(result.reason_codes) == {
        "scope_days_insufficient:daily",
        "scope_certificate_missing:ths_hot",
        "unauthorized:ths_hot",
    }
    assert not ProductReadiness("P02", "incomplete", ("x",), {}).ready
    with pytest.raises(ValueError, match="invalid_scope_requirement"):
        ScopeRequirement("", -1)


def test_readiness_requires_explicit_partition_count_for_partitioned_catalogs():
    """场景：主题成员只完成一个分区；预期：即使证书为 certified 也不能满足 1666 分区。"""
    item = certificate("ths_member") | {"completed_partition_count": 1}
    result = assess_product_readiness(
        "P02",
        (item,),
        (ScopeRequirement("ths_member", minimum_completed_partitions=1666),),
    )
    assert result.reason_codes == ("scope_partitions_insufficient:ths_member",)
