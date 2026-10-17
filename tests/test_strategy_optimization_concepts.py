"""概念历史附加层的同日成员、覆盖、缺失和独立读取契约。"""

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from tushare_entitlement_hub.strategy_optimization import cli, concepts
from tushare_entitlement_hub.strategy_optimization.strategies import strategy_grid


def concept_fixture(root):
    """输入临时目录，输出七只股票两日特征及同日概念，首日无历史、次日全部上涨。"""
    source, raw = root / "source", root / "raw"
    source.mkdir()
    raw.mkdir()
    codes = [f"{index:06}.SH" for index in range(7)]
    (source / "dataset.json").write_text(
        json.dumps(
            {
                "calendar": ["20241219", "20241220"],
                "securities": codes,
                "source_fingerprint": "synthetic",
            }
        )
    )
    for code in codes:
        np.savez(source / f"{code}.npz", return1=np.array([0.01, 0.02]))
    collection = {
        "selection": "same_day_top5",
        "days": [{"trade_date": "20241220", "chosen": ["A", "B"]}],
    }
    (raw / "collection.json").write_text(json.dumps(collection))
    for concept, members in [("A", codes), ("B", codes[:1])]:
        (raw / f"members-20241220-{concept}.json").write_text(
            json.dumps(
                {
                    "response": {
                        "code": 0,
                        "data": {
                            "fields": ["trade_date", "ts_code", "con_code"],
                            "items": [["20241220", concept, code] for code in members],
                        },
                    }
                }
            )
        )
    return source, raw, codes


def test_concept_overlay_keeps_history_and_excludes_self(tmp_path):
    """场景：七只上涨股票共享概念；预期：当日群动通过、前日未知、基础文件复用。"""
    source, raw, codes = concept_fixture(tmp_path)
    output = tmp_path / "joined"
    path = concepts.attach_concepts(source, raw, output)
    assert json.loads(path.read_text())["concept_history_days"] == 1
    features = concepts.load_features(output, codes[0])
    np.testing.assert_array_equal(features["concept_group"], [False, True])
    assert (output / f"{codes[0]}.npz").is_symlink()
    assert "concept_group" not in concepts.load_features(source, codes[0])
    with pytest.raises(ValueError, match="output_exists"):
        concepts.attach_concepts(source, raw, output)
    assert len(strategy_grid(include_concepts=True)) == 96


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "foreign_day", "member"])
def test_concepts_refuse_incomplete_or_misaligned_history(tmp_path, mutation):
    """场景：漏日、重复日、区间外日期或跨日成员；预期：拒绝创建完整数据集清单。"""
    source, raw, _ = concept_fixture(tmp_path)
    path = raw / "collection.json"
    collection = json.loads(path.read_text())
    if mutation == "missing":
        collection["days"] = []
    elif mutation == "duplicate":
        collection["days"] *= 2
    elif mutation == "foreign_day":
        collection["days"][0]["trade_date"] = "19000101"
    else:
        member = raw / "members-20241220-A.json"
        member.write_text(member.read_text().replace("20241220", "20241221"))
    path.write_text(json.dumps(collection))
    with pytest.raises(ValueError, match="concept_"):
        concepts.attach_concepts(source, raw, tmp_path / "output")
    assert not (tmp_path / "output/dataset.json").exists()


def test_concept_command_passes_paths(tmp_path, monkeypatch):
    """场景：离线概念附加CLI；预期：不连接网络且输出已创建清单路径。"""
    monkeypatch.setattr(cli, "attach_concepts", lambda *args: tmp_path / "dataset.json")
    result = CliRunner().invoke(cli.optimization_app, ["attach-concepts", "base", "raw", "out"])
    assert result.exit_code == 0
    assert "dataset.json" in result.output
