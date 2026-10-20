"""微型真实DuckDB与确定性特征验证数据构建、全样本搜索及全部尝试记录。"""

import json
from pathlib import Path

import duckdb
import numpy as np
import pytest

from tushare_entitlement_hub.strategy_optimization import dataset, search
from tushare_entitlement_hub.strategy_optimization.execution import ExitPolicy
from tushare_entitlement_hub.strategy_optimization.strategies import StrategyDefinition


def envelope(path: Path, rows: list[dict]) -> None:
    """写无凭据的标准响应；输入一行字典，输出fields/items公开包络文件。"""
    fields = list(rows[0]) if rows else ["unused"]
    path.write_text(
        json.dumps(
            {
                "response": {
                    "code": 0,
                    "data": {
                        "fields": fields,
                        "items": [[row[name] for name in fields] for row in rows],
                    },
                }
            }
        )
    )


def miniature_database(path: Path, calendar: list[str], *, empty_manifest=False) -> None:
    """创建真实只读入口所需清单；输入三日历，输出已认证合成源范围，业务行由测试注入。"""
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE TABLE cache_manifest(manifest JSON)")
        if not empty_manifest:
            connection.execute(
                "INSERT INTO cache_manifest VALUES (?)",
                [
                    json.dumps(
                        {
                            "raw_fingerprint": "synthetic-public-input",
                            "trading_dates": calendar,
                            "partition_coverage": {
                                name: {
                                    "status": "complete",
                                    "missing_dates": [],
                                    "passed_dates": calendar,
                                }
                                for name in ("daily", "adj_factor", "stk_limit")
                            },
                        }
                    )
                ],
            )


def test_build_dataset_aggregates_other_members_and_reuses_complete_cache(tmp_path, monkeypatch):
    """场景：250只合成同业含无历史身份证券；预期：剔除本股、记录北交所排除且可重用。"""
    calendar = ["20230101", "20230102", "20230103"]
    database_path = tmp_path / "source.duckdb"
    miniature_database(database_path, calendar)
    root = tmp_path / "sectors"
    root.mkdir()
    envelope(root / "catalog.json", [{"index_code": "X"}])
    codes = [f"{index:06}.SZ" for index in range(250)]
    envelope(
        root / "members-X-Y.json",
        [
            dict(ts_code=code, l1_code="X", in_date="20220101", out_date=None, is_new="Y")
            for code in codes
        ],
    )
    envelope(
        root / "daily-X.json", [dict(trade_date=day, close=10.0, amount=100.0) for day in calendar]
    )

    def security_history(connection):
        """输入只读连接，输出250只相同路径及一只北交所；末日涨价便于手算群动。"""
        for code in codes + ["920001.BJ", "UNMAPPED.SH"]:
            yield (
                code,
                [
                    dict(
                        trade_date=day,
                        open=10.0,
                        high=12.0,
                        low=9.0,
                        close=10.0 + index,
                        vol=100.0,
                        amount=100.0,
                        adj_factor=1.0,
                        up_limit=13.0,
                        down_limit=8.0,
                    )
                    for index, day in enumerate(calendar)
                ],
            )

    monkeypatch.setattr(dataset, "iter_security_history", security_history)
    output = tmp_path / "features"
    path = dataset.build_dataset(database_path, root, output)
    manifest = json.loads(path.read_text())
    assert manifest["historical_pit_certified"] is False
    assert manifest["membership_diagnostics"]["excluded_bj_securities"] == 1
    assert len(manifest["securities"]) == 251
    with np.load(output / "000000.SZ.npz") as saved:
        assert saved["sector_peers"][-1] == 249
        assert saved["sector_breadth"][-1] == 1
        assert saved["sector_mean_return"][-1] == pytest.approx(12 / 11 - 1)
    assert dataset.build_dataset(database_path, root, output) == path
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "partial").touch()
    with pytest.raises(ValueError, match="incomplete_dataset"):
        dataset.build_dataset(database_path, root, incomplete)
    missing = tmp_path / "missing.duckdb"
    miniature_database(missing, calendar, empty_manifest=True)
    with pytest.raises(ValueError, match="history_manifest_missing"):
        dataset.build_dataset(missing, root, tmp_path / "bad")


@pytest.mark.parametrize("accelerate", [False, True])
def test_search_persists_all_trials_censoring_and_real_frequency(tmp_path, monkeypatch, accelerate):
    """场景：微型完整日期边界、盈利/亏损/未退出和到期；预期：全部试验留存，不伪装正式验收。"""
    source = tmp_path / "features"
    source.mkdir()
    calendar = [
        "20200101",
        "20211018",
        "20220101",
        "20230101",
        "20240101",
        "20250101",
        "20260101",
        "20261015",
        "20261016",
    ]
    (source / "dataset.json").write_text(
        json.dumps(
            {
                "calendar": calendar,
                "securities": ["A.SH"],
                "source_fingerprint": "synthetic",
                **({"concept_root": "synthetic"} if accelerate else {}),
            }
        )
    )
    np.savez(source / "A.SH.npz", valid=np.ones(9))
    definitions = (
        StrategyDefinition("macd_event", "concept_hot"),
        StrategyDefinition("range_bottom", "concept_group"),
    )
    monkeypatch.setattr(search, "strategy_grid", lambda **kwargs: definitions)
    monkeypatch.setattr(search, "exit_grid", lambda: (ExitPolicy(0.025, 0.06, 20),))
    monkeypatch.setattr(
        search,
        "strategy_masks",
        lambda *_: {
            definitions[0].name: np.ones(9, dtype=bool),
            definitions[1].name: np.zeros(9, dtype=bool),
        },
    )
    monkeypatch.setattr(search, "entry_indices", lambda *_: np.arange(9))
    outcomes = {
        "entry_index": np.array([1, 2, 3, -1, 5, -1, 7, -1, -1]),
        "exit_index": np.array([2, 3, 4, 9, 6, 9, 9, 9, 9]),
        "return": np.array([0.01, 0.03, -0.02, np.nan, 0.04, np.nan, np.nan, np.nan, np.nan]),
    }
    monkeypatch.setattr(search, "outcome_arrays", lambda *_: outcomes)
    from tushare_entitlement_hub.strategy_optimization.acceleration import accumulate_policy

    monkeypatch.setattr(search, "optional_compiled_accumulator", lambda: accumulate_policy)
    monkeypatch.setattr(search, "entry_indices", lambda *_: outcomes["entry_index"])
    output = tmp_path / "result"
    assert search.run_search(
        source, output, maximum_securities=1, accelerate=accelerate, concepts_only=accelerate
    ).exists()
    result = json.loads((output / "search.json").read_text())
    assert result["trials"] == 2
    assert result["target_passed_trials"] == 0
    assert any(row["open"] for row in result["best"])
    assert any(row["closed"] == 0 for row in result["best"])
    with pytest.raises(ValueError, match="search_output_exists"):
        search.run_search(source, output)
    with pytest.raises(ValueError, match="maximum_securities"):
        search.run_search(source, tmp_path / "invalid", maximum_securities=0)


def test_concept_search_requires_daily_history_overlay(tmp_path):
    """场景：基础特征没有历史概念附加层；预期：概念专属搜索立即拒绝，不生成假全覆盖。"""
    source = tmp_path / "source"
    source.mkdir()
    (source / "dataset.json").write_text(json.dumps({"calendar": [], "securities": []}))
    with pytest.raises(ValueError, match="concept_search_requires"):
        search.run_search(source, tmp_path / "output", concepts_only=True)
