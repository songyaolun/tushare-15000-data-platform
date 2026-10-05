"""完整参考库、绑定和产品落盘链的逐接口验收。"""

from pathlib import Path

from reference_chain_fixture import run_reference_chain


def test_all_products_strict_reference_chain(tmp_path: Path) -> None:
    """206 个来源实际读取并执行；新增 107 个来源有自身证据或明确失败原因。

    例如缺单位的真实字段保持 unknown，不以人工百分比单位或证书补齐 observed。
    所有链路产物写入固定子目录，供当前测试轮次按 basetemp 归档。
    """
    summary = run_reference_chain(tmp_path / "strict-reference-chain")
    assert summary["valid"]
    assert summary["generated_reference_apis"] == 206
    assert summary["new_api_states"]["own_product_reference"] == 107
    assert summary["new_no_own_product_reference"] == []
    for state in ("readable", "executed"):
        assert summary["all_api_states"][state] == 206
        assert summary["new_api_states"][state] == 107
    assert summary["all_api_states"]["configured"] == 205
    assert summary["new_api_states"]["configured"] == 107
