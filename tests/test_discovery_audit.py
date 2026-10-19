"""独立CSV审计的合成账本与故意破坏测试，不调用模拟器和组合评价器。"""

import csv
import json
import math

import pytest

from tushare_entitlement_hub.strategy_discovery.audit import audit_experiments, audit_result


def write_rows(path, rows, fields):
    """保存测试CSV并保留空表表头；例空rows和['day']生成只有day表头的文件。"""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_fixture(path, *, empty=False):
    """生成四日跨年账本；例默认输出两笔费用后收益-.1/.2、末端NAV1089000。"""
    path.mkdir(parents=True)
    daily = [
        {"day": day, "date": date, "nav": nav, "cash": 500000, "exposure": 0.5, "return": change}
        for day, date, nav, change in zip(
            range(10, 14),
            ["20241230", "20241231", "20250102", "20250103"],
            [900000, 990000, 990000, 1089000],
            [-0.1, 0.1, 0, 0.1],
            strict=True,
        )
    ]
    trades = (
        []
        if empty
        else [
            {"entry": 10, "exit": 12, "cost": 100, "proceeds": 89, "dividend": 1, "return": -0.1},
            {"entry": 11, "exit": 13, "cost": 100, "proceeds": 119, "dividend": 1, "return": 0.2},
        ]
    )
    entries = [] if empty else [{"day": 10}, {"day": 11}, {"day": 11}]
    metrics = {
        "total_return": 0.089,
        "annualized_return": 1.089 ** (250 / 4) - 1,
        "maximum_drawdown": -0.1,
        "closed_trades": len(trades),
        "win_rate": None if empty else 0.5,
        "mean_trade_return": None if empty else 0.05,
        "maximum_entry_gap": 4 if empty else 2,
        "entry_count": len(entries),
        "realized_profit": 0 if empty else 10,
        "reconciliation_error": 0,
        "mean_exposure": 0.5,
        "terminal_nav_if_stale_zero": 1088000,
        "stale_value": 1000,
        "annual": {
            "2024": {
                "total_return": -0.01,
                "annualized_return": 0.99**125 - 1,
                "maximum_drawdown": -0.1,
                "trades_closed": 0,
                "win_rate": None,
                "entries": len(entries),
            },
            "2025": {
                "total_return": 0.1,
                "annualized_return": 1.1**125 - 1,
                "maximum_drawdown": 0,
                "trades_closed": len(trades),
                "win_rate": None if empty else 0.5,
                "entries": 0,
            },
        },
    }
    save_fixture(path, metrics, daily, trades, entries)
    return metrics, daily, trades, entries


def save_fixture(path, metrics, daily, trades, entries):
    """重写合成导出便于逐项破坏；例修改daily[0]['cash']后保存可测试负现金拒绝。"""
    (path / "metrics.json").write_text(json.dumps({"configuration": {}, "metrics": metrics}))
    write_rows(path / "daily.csv", daily, ["day", "date", "nav", "cash", "exposure", "return"])
    write_rows(
        path / "trades.csv", trades, ["entry", "exit", "cost", "proceeds", "dividend", "return"]
    )
    write_rows(path / "entries.csv", entries, ["day"])


def test_independent_audit_reconstructs_cross_year_and_fee_adjusted_results(tmp_path):
    """场景：首日亏损、跨年回升及含分红的净 proceeds；预期：正确重算而不漏初始峰值。"""
    path = tmp_path / "strategy_a"
    audit_fixture(path)
    result = audit_result(path)
    assert result["name"] == "strategy_a"
    assert result["terminal_nav"] == 1089000
    assert result["maximum_drawdown"] == pytest.approx(-0.1)
    assert result["closed_trades"] == 2
    assert result["mean_trade_return"] == pytest.approx(0.05)
    assert result["maximum_entry_gap"] == 2
    assert result["annual"]["2024"]["total_return"] == pytest.approx(-0.01)
    assert result["annual"]["2025"]["total_return"] == pytest.approx(0.1)


def test_explicit_capital_reconstructs_small_account_and_rejects_wrong_default(tmp_path):
    """场景：12万元合成账本；预期显式本金正确复算，误用100万基准被拒绝。"""
    path = tmp_path / "small_account"
    metrics, daily, trades, entries = audit_fixture(path)
    for row in daily:
        row["nav"] *= 0.12
        row["cash"] *= 0.12
    metrics["terminal_nav_if_stale_zero"] = daily[-1]["nav"] - metrics["stale_value"]
    save_fixture(path, metrics, daily, trades, entries)
    result = audit_result(path, initial_capital=120000)
    assert result["total_return"] == pytest.approx(0.089)
    assert result["terminal_nav"] == pytest.approx(130680)
    with pytest.raises(ValueError, match=r"daily\[0\].return"):
        audit_result(path)


@pytest.mark.parametrize("capital", [0, -1, True, float("nan"), float("inf")])
def test_invalid_explicit_initial_capital_fails_before_reading_account(tmp_path, capital):
    """场景：非正或非有限本金；预期明确拒绝，不推测或读取缺失账户。"""
    with pytest.raises(ValueError, match="initial_capital"):
        audit_result(tmp_path / "account", initial_capital=capital)


@pytest.mark.parametrize("zero_bytes", [False, True])
def test_zero_counts_accept_header_only_or_zero_byte_exports(tmp_path, zero_bytes):
    """场景：零交易零入场的两种既有导出格式；预期：胜率未知、全区间空窗为4日。"""
    path = tmp_path / "empty_trades"
    audit_fixture(path, empty=True)
    if zero_bytes:
        (path / "trades.csv").write_text("")
        (path / "entries.csv").write_text("")
    result = audit_result(path)
    assert result["win_rate"] is None
    assert result["mean_trade_return"] is None
    assert result["maximum_entry_gap"] == 4


@pytest.mark.parametrize(
    "field",
    [
        "total_return",
        "annualized_return",
        "maximum_drawdown",
        "closed_trades",
        "win_rate",
        "mean_trade_return",
        "maximum_entry_gap",
        "mean_exposure",
        "entry_count",
        "realized_profit",
        "reconciliation_error",
        "terminal_nav_if_stale_zero",
    ],
)
def test_corrupted_scalar_metrics_are_rejected_with_configuration_and_field(tmp_path, field):
    """场景：任一汇总指标被改写；预期：报错包含具体配置与指标，不输出合格结果。"""
    path = tmp_path / "broken_strategy"
    metrics, *rows = audit_fixture(path)
    metrics[field] += 0.1
    save_fixture(path, metrics, *rows)
    with pytest.raises(ValueError, match=f"broken_strategy:{field}"):
        audit_result(path)


@pytest.mark.parametrize(
    ("field", "value", "error_field"),
    [
        ("nav", 0, "nav"),
        ("cash", -1, "cash"),
        ("exposure", 1.01, "exposure"),
        ("exposure", -0.01, "exposure"),
        ("return", 0.9, "return"),
        ("nav", "bad", "nav"),
        ("nav", float("nan"), "nav"),
        ("day", 10.5, "day"),
        ("day", -1, "day"),
        ("date", "20240231", "date"),
    ],
)
def test_bad_daily_values_fail_closed(tmp_path, field, value, error_field):
    """场景：非有限/非数值净值、负现金、杠杆、错收益或非法日期；预期：逐字段拒绝。"""
    path = tmp_path / "bad_daily"
    metrics, daily, trades, entries = audit_fixture(path)
    daily[0][field] = value
    save_fixture(path, metrics, daily, trades, entries)
    with pytest.raises(ValueError, match=error_field):
        audit_result(path)


@pytest.mark.parametrize(("field", "value"), [("day", 10), ("date", "20241230")])
def test_duplicate_days_or_dates_cannot_hide_missing_history(tmp_path, field, value):
    """场景：第二条记录重复日期或日编号；预期：拒绝被压缩或重复的日序列。"""
    path = tmp_path / "bad_sequence"
    metrics, daily, trades, entries = audit_fixture(path)
    daily[1][field] = value
    save_fixture(path, metrics, daily, trades, entries)
    with pytest.raises(ValueError, match="daily"):
        audit_result(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("entry", 9),
        ("entry", 10.5),
        ("exit", 10),
        ("exit", 20),
        ("cost", 0),
        ("proceeds", -1),
        ("return", 0.5),
        ("dividend", 10),
    ],
)
def test_trade_dates_cashflows_and_after_fee_returns_are_checked(tmp_path, field, value):
    """场景：交易跨出研究区间、零成本、负卖款或费用后收益不符；预期：交易审计失败。"""
    path = tmp_path / "bad_trade"
    metrics, daily, trades, entries = audit_fixture(path)
    trades[0][field] = value
    save_fixture(path, metrics, daily, trades, entries)
    with pytest.raises(ValueError, match="trades"):
        audit_result(path)


@pytest.mark.parametrize(
    "mutation",
    [
        "annual_year",
        "annual_fields",
        "annual_value",
        "missing_metric",
        "invalid_metric_type",
        "boolean_metric",
        "negative_stale",
        "oversized_stale",
        "entry_outside",
        "null_win_rate",
    ],
)
def test_structural_and_annual_metrics_cannot_silently_skip_checks(tmp_path, mutation):
    """场景：年度缺项、非法类型、缺指标或非法入场；预期：严格失败而不跳过缺项。"""
    path = tmp_path / "bad_structure"
    metrics, daily, trades, entries = audit_fixture(path, empty=mutation == "null_win_rate")
    if mutation == "annual_year":
        del metrics["annual"]["2024"]
    elif mutation == "annual_fields":
        del metrics["annual"]["2024"]["total_return"]
    elif mutation == "annual_value":
        metrics["annual"]["2024"]["total_return"] = 0
    elif mutation == "missing_metric":
        del metrics["total_return"]
    elif mutation == "invalid_metric_type":
        metrics["total_return"] = {}
    elif mutation == "boolean_metric":
        metrics["total_return"] = True
    elif mutation == "negative_stale":
        metrics["stale_value"] = -1
    elif mutation == "oversized_stale":
        metrics["stale_value"] = 2000000
    elif mutation == "entry_outside":
        entries[0]["day"] = 999
    else:
        metrics["win_rate"] = 0
    save_fixture(path, metrics, daily, trades, entries)
    with pytest.raises(ValueError, match="audit_failed:bad_structure"):
        audit_result(path)


@pytest.mark.parametrize("filename", ["daily.csv", "trades.csv", "entries.csv", "metrics.json"])
def test_missing_evidence_file_fails(tmp_path, filename):
    """场景：任一必要证据文件不存在；预期：指出配置和缺失文件。"""
    path = tmp_path / "missing_file"
    audit_fixture(path)
    (path / filename).unlink()
    with pytest.raises(ValueError, match=filename):
        audit_result(path)


@pytest.mark.parametrize("mutation", ["empty", "columns", "duplicate", "extra", "short", "rows"])
def test_missing_columns_or_malformed_csv_fail(tmp_path, mutation):
    """场景：空文件、缺列、重复列、长短行或仅表头；预期：拒绝不完整日证据。"""
    path = tmp_path / "bad_csv"
    audit_fixture(path)
    source = path / "daily.csv"
    lines = source.read_text().splitlines()
    if mutation == "empty":
        lines = []
    elif mutation == "columns":
        lines[0] = lines[0].replace("nav", "unknown")
    elif mutation == "duplicate":
        lines[0] += ",nav"
    elif mutation == "extra":
        lines[1] += ",extra"
    elif mutation == "short":
        lines[1] = "10"
    else:
        lines = lines[:1]
    source.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="daily"):
        audit_result(path)


@pytest.mark.parametrize("filename", ["trades.csv", "entries.csv"])
def test_nonzero_counts_reject_zero_byte_export(tmp_path, filename):
    """场景：有交易/入场计数却导出零字节；预期：拒绝把缺失数据解释为零记录。"""
    path = tmp_path / "bad_empty"
    audit_fixture(path)
    (path / filename).write_text("")
    with pytest.raises(ValueError, match=filename):
        audit_result(path)


@pytest.mark.parametrize("content", ["not json", "[]", '{"metrics": []}'])
def test_malformed_json_is_rejected(tmp_path, content):
    """场景：metrics不是合法对象；预期：带配置定位的JSON审计失败。"""
    path = tmp_path / "bad_json"
    audit_fixture(path)
    (path / "metrics.json").write_text(content)
    with pytest.raises(ValueError, match="bad_json:metrics.json"):
        audit_result(path)


def test_experiment_summary_counts_all_passed_configurations(tmp_path):
    """场景：实验根目录及另一单配置同时输入；预期：汇总三配置四交易并保存JSON。"""
    root = tmp_path / "experiment"
    audit_fixture(root / "a")
    audit_fixture(root / "b", empty=True)
    standalone = tmp_path / "standalone"
    audit_fixture(standalone)
    (root / "experiment.json").write_text("{}")
    (root / "attempts.csv").write_text("name\na\nb\n")
    output = tmp_path / "reports" / "audit.json"
    assert audit_experiments([root, standalone], output) == output
    summary = json.loads(output.read_text())
    assert summary["passed_configurations"] == 3
    assert summary["closed_trades"] == 4
    assert [item["name"] for item in summary["configurations"]] == ["a", "b", "standalone"]


@pytest.mark.parametrize("mutation", ["empty", "missing", "no_configs", "duplicate", "bad_child"])
def test_invalid_experiments_never_write_pass_report(tmp_path, mutation):
    """场景：空输入、根不存在、无配置、重复路径或配置损坏；预期：不写通过报告。"""
    root = tmp_path / "experiment"
    audit_fixture(root / "a")
    roots = [root]
    if mutation == "empty":
        roots = []
    elif mutation == "missing":
        roots = [tmp_path / "missing"]
    elif mutation == "no_configs":
        empty_root = tmp_path / "empty"
        empty_root.mkdir()
        roots = [empty_root]
    elif mutation == "duplicate":
        roots.append(root / "a")
    else:
        (root / "a" / "metrics.json").unlink()
    output = tmp_path / "audit.json"
    with pytest.raises(ValueError, match="audit_failed"):
        audit_experiments(roots, output)
    assert not output.exists()


def test_standalone_csv_without_metrics_is_not_skipped(tmp_path):
    """场景：直接传入缺metrics的配置目录；预期：审计该配置失败而非忽略目录。"""
    path = tmp_path / "incomplete"
    audit_fixture(path)
    (path / "metrics.json").unlink()
    with pytest.raises(ValueError, match="metrics.json"):
        audit_experiments([path], tmp_path / "audit.json")


def test_tiny_rounding_errors_are_tolerated(tmp_path):
    """场景：序列化带来1e-10量级收益和对账误差；预期：按1e-8容忍通过。"""
    path = tmp_path / "rounding"
    metrics, *rows = audit_fixture(path)
    metrics["total_return"] += 1e-10
    metrics["reconciliation_error"] = 1e-10
    save_fixture(path, metrics, *rows)
    assert audit_result(path)["closed_trades"] == 2


@pytest.mark.parametrize("scale", [1, 400])
@pytest.mark.parametrize("sign", [-1, 1])
def test_reconciliation_allows_balance_scaled_float_roundoff(tmp_path, sign, scale):
    """场景：百万或数亿资产重构相差55个浮点间距；预期保留误差且通过核账。"""
    path = tmp_path / "balance_roundoff"
    metrics, daily, trades, entries = audit_fixture(path)
    for row in daily:
        row["nav"] *= scale
        row["cash"] *= scale
    metrics["stale_value"] *= scale
    metrics["terminal_nav_if_stale_zero"] *= scale
    metrics["reconciliation_error"] = sign * 55 * math.ulp(daily[-1]["nav"])
    save_fixture(path, metrics, daily, trades, entries)
    assert audit_result(path, initial_capital=1_000_000 * scale)["closed_trades"] == 2


@pytest.mark.parametrize("scale,error", [(400, "65ulp"), (1_000_000, 1.1e-5), (400, .01)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_large_balance_reconciliation_keeps_ulp_and_absolute_limits(tmp_path, scale, error, sign):
    """场景：大账户差额越过64间距或十万分之一元；预期双向拒绝，分币差错也拒绝。"""
    path = tmp_path / "large_balance_error"
    metrics, daily, trades, entries = audit_fixture(path)
    for row in daily:
        row["nav"] *= scale
        row["cash"] *= scale
    metrics["stale_value"] *= scale
    metrics["terminal_nav_if_stale_zero"] *= scale
    metrics["reconciliation_error"] = sign * (
        65 * math.ulp(daily[-1]["nav"]) if error == "65ulp" else error
    )
    save_fixture(path, metrics, daily, trades, entries)
    with pytest.raises(ValueError, match="reconciliation_error"):
        audit_result(path, initial_capital=1_000_000 * scale)


@pytest.mark.parametrize("error", [0.0000011, -0.0000011, 0.01, -0.01])
def test_reconciliation_rejects_errors_above_roundoff_ceiling(tmp_path, error):
    """场景：重构损益差超过百万分之一元或达到分币；预期仍拒绝真实差额。"""
    path = tmp_path / "material_reconciliation"
    metrics, *rows = audit_fixture(path)
    metrics["reconciliation_error"] = error
    save_fixture(path, metrics, *rows)
    with pytest.raises(ValueError, match="reconciliation_error"):
        audit_result(path)


def test_entry_gap_includes_initial_boundary_before_first_entry(tmp_path):
    """场景：四日研究只在第三日首次入场且未平仓；预期：前端两日空窗纳入最长间隔。"""
    path = tmp_path / "late_first_entry"
    metrics, daily, trades, _ = audit_fixture(path, empty=True)
    entries = [{"day": 12}]
    metrics["entry_count"] = 1
    metrics["maximum_entry_gap"] = 2
    metrics["annual"]["2025"]["entries"] = 1
    save_fixture(path, metrics, daily, trades, entries)
    assert audit_result(path)["maximum_entry_gap"] == 2
