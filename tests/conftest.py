"""全测试会话离线，fixture 只含确定性公开示例。"""

import socket

import pytest

from tushare_entitlement_hub.models import CatalogEntry, Policy
from tushare_entitlement_hub.store import Store


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    """场景：任何真实网络连接；预期：立即拒绝。例：connect(address) → AssertionError。"""

    def denied(*args, **kwargs):
        """拒绝底层连接；输入任何地址，输出固定离线测试错误。"""
        raise AssertionError("离线测试禁止真实网络")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


@pytest.fixture
def policy():
    """场景：完整指数政策；预期：没有政策债务。例：fixture → index_daily 政策。"""
    return Policy(
        "index_daily",
        category="index",
        lifecycle="production",
        consumers=["market"],
        primary_key=["ts_code", "trade_date"],
        date_field="trade_date",
        shard_by="trade_date",
        required_fields=["ts_code", "trade_date", "close"],
        units={"close": "元"},
        collection_mode="daily",
        purpose="展示指数行情",
        page_size=10,
    )


@pytest.fixture
def entry(policy):
    """场景：文档完整的积分接口；预期：可以制定计划。例：fixture → production 目录行。"""
    return CatalogEntry(
        policy.api_name,
        title="指数行情",
        doc_url="https://tushare.pro/document/2?doc_id=95",
        category="index",
        permission_mode="points",
        minimum_points=2000,
        updated_at="每日收盘后",
        last_verified="2024-01-03T00:00:00Z",
        lifecycle="production",
        policy=policy,
    )


@pytest.fixture
def store(tmp_path):
    """场景：隔离本地数据库；预期：退出后释放连接。例：tmp_path → 空 Store。"""
    database = Store(tmp_path / "hub.duckdb")
    yield database
    database.close()
