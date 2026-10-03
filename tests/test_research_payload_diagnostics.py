"""单结果8MiB门禁的内部定位字段与CLI固定错误边界，不输出载荷。"""

import json
from dataclasses import replace

import pytest
from test_research_governance import inputs
from typer.testing import CliRunner

from tushare_entitlement_hub.cli import app
from tushare_entitlement_hub.equity_research import cli, storage
from tushare_entitlement_hub.equity_research.common import make_result
from tushare_entitlement_hub.equity_research.consumption_registry import PRODUCT_ALIASES
from tushare_entitlement_hub.equity_research.models import MARKET_ENTITY
from tushare_entitlement_hub.store import encode


def test_real_oversized_result_identifies_product_entity_bytes_before_sql_and_cli_stays_safe(
    monkeypatch, tmp_path
):
    """真实P06市场result超过8MiB时先失败；异常属性有定位字段，字符串和CLI不泄露载荷或属性。"""
    context = inputs()
    product = PRODUCT_ALIASES["P06"]
    original = make_result(context, product, MARKET_ENTITY)
    result = replace(
        original,
        details_json=encode(
            {
                "marker": "SYNTHETIC_PAYLOAD_SENTINEL",
                "padding": "中" * (storage.MAXIMUM_RESULT_JSON_BYTES // 3),
            }
        ),
    )
    expected_bytes = len(result.to_json().encode())
    assert expected_bytes > 8 * 1024 * 1024 == storage.MAXIMUM_RESULT_JSON_BYTES

    class NoSQL:
        """容量错误必须在任何DuckDB写入前发生；无需分配数据库内存即可验证。"""

        def execute(self, *args, **kwargs):
            """SQL被调用即失败，确保报错不是由DuckDB解析超大JSON后才触发。"""
            pytest.fail("容量门禁前调用了SQL")

    with pytest.raises(storage.ResultPayloadCapacityError) as captured:
        storage.save_results(NoSQL(), (result,), 128)
    error = captured.value
    assert vars(error) == {
        "product_id": product,
        "entity_id": MARKET_ENTITY,
        "payload_bytes": expected_bytes,
    }
    assert str(error) == storage.RESULT_PAYLOAD_CAPACITY_ERROR
    assert "SYNTHETIC_PAYLOAD_SENTINEL" not in repr(error)

    def failed(*args, **kwargs):
        """向CLI传递真实门禁异常对象，验证其内部诊断属性不会被公开打印。"""
        raise error

    monkeypatch.setattr(cli, "run_research", failed)
    response = CliRunner().invoke(
        app,
        [
            "research",
            "run",
            "--reference",
            str(tmp_path / "ref"),
            "--output",
            str(tmp_path / "out"),
            "--as-of",
            context.analysis_as_of,
            "--product",
            "P06",
        ],
    )
    assert response.exit_code == 1
    assert json.loads(response.stdout) == {
        "status": "failed",
        "reason_codes": [storage.RESULT_PAYLOAD_CAPACITY_ERROR],
    }
    assert response.stderr == ""
    assert "SYNTHETIC_PAYLOAD_SENTINEL" not in response.output
    assert product not in response.output and MARKET_ENTITY not in response.output


@pytest.mark.parametrize(
    "payload", [[], {}, {"envelope": []}, {"envelope": {"product_id": 1, "entity_id": None}}]
)
def test_nonmodel_payload_diagnostics_never_copy_arbitrary_fields(payload):
    """底层通用insert输入缺少合法身份时诊断记None；只保留三个允许属性和固定错误码。"""
    error = storage.ResultPayloadCapacityError(payload, 9000000)
    assert vars(error) == {"product_id": None, "entity_id": None, "payload_bytes": 9000000}
    assert str(error) == storage.RESULT_PAYLOAD_CAPACITY_ERROR
