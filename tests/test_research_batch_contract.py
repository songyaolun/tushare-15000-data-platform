"""研究库与更新后的共享批处理上限保持一致，拒绝任意大内存读取。"""

from pathlib import Path

import pytest

from tushare_entitlement_hub.equity_research.models import ResearchConfig
from tushare_entitlement_hub.equity_research.source_validation import verify_source_evidence
from tushare_entitlement_hub.equity_research.storage import save_results, stored_rows, verify_run


class RejectDatabaseAccess:
    """非法批大小应在任何SQL前拒绝，访问本替身即测试失败。"""

    def execute(self, *arguments, **keywords):
        """阻止过早数据库访问；例：execute('SELECT 1')立即抛AssertionError。"""
        raise AssertionError("非法批大小不能访问数据库")


@pytest.mark.parametrize("batch_size", [0, -1, True, 513, 1000000])
@pytest.mark.parametrize("entry", ["save", "read", "verify", "evidence"])
def test_research_entry_points_share_bounded_batch_contract(batch_size, entry):
    """场景：四个直接入口收到非法批大小；预期：在读库或消费生成器前按统一契约拒绝。"""
    connection = RejectDatabaseAccess()

    def untouched_results():
        """模拟不能提前消费的输入；例：开始迭代时失败，合法门禁应使它保持未执行。"""
        raise AssertionError("非法批大小不能消费输入")
        yield

    with pytest.raises(ValueError, match="invalid_batch_size"):
        if entry == "save":
            save_results(connection, untouched_results(), batch_size)
        elif entry == "read":
            list(stored_rows(connection, "run", batch_size))
        elif entry == "verify":
            verify_run(connection, {}, batch_size=batch_size)
        else:
            verify_source_evidence(
                Path("absent.duckdb"), connection, "run", ResearchConfig(), batch_size=batch_size
            )
