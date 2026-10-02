"""派生参考数据只使用已发布本地证据，不访问网络或改写来源表。"""

import duckdb

from tushare_entitlement_hub.reference_audit import audit_reference
from tushare_entitlement_hub.reference_derived import publish_derived_datasets


def test_equity_calendar_preserves_official_rows_and_derives_bse_evidence():
    """场景：上深日历一致且某日有北交所行情；预期：原始日历保留，北交所行带证据等级。"""
    with duckdb.connect() as connection:
        connection.execute('CREATE SCHEMA accepted')
        connection.execute(
            "CREATE TABLE accepted.trade_cal AS SELECT * FROM (VALUES "
            "('SSE', DATE '2026-09-30', '1'), ('SZSE', DATE '2026-09-30', '1'), "
            "('SSE', DATE '2026-10-01', '0'), ('SZSE', DATE '2026-10-01', '0')) "
            "t(exchange, cal_date, is_open)"
        )
        connection.execute(
            "CREATE TABLE accepted.daily AS SELECT '920001.BJ' ts_code, "
            "DATE '2026-09-30' trade_date"
        )
        assert publish_derived_datasets(connection)
        rows = connection.execute(
            "SELECT exchange, cal_date, is_open, evidence_basis, observed_security_count "
            "FROM derived.cn_equity_trade_calendar ORDER BY exchange, cal_date"
        ).fetchall()
    assert ('BSE', rows[0][1], True, 'observed_bse_market_activity', 1) in rows
    assert ('BSE', rows[1][1], False, 'aligned_sse_szse_calendar', 0) in rows
    assert sum(row[0] in {'SSE', 'SZSE'} and row[3] == 'official_exchange_calendar'
               for row in rows) == 4


def test_equity_calendar_marks_disagreement_unresolved_and_handles_missing_tables():
    """场景：上深开闭市不一致或来源表缺失；预期：北交所日期不猜测，缺表时不创建产品表。"""
    with duckdb.connect() as connection:
        connection.execute('CREATE SCHEMA accepted')
        assert not publish_derived_datasets(connection)
        assert connection.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema='derived' AND table_name='cn_equity_trade_calendar'"
        ).fetchone()[0] == 0
        connection.execute(
            "CREATE TABLE accepted.trade_cal AS SELECT * FROM (VALUES "
            "('SSE', DATE '2026-09-30', '1'), ('SZSE', DATE '2026-09-30', '0')) "
            "t(exchange, cal_date, is_open)"
        )
        connection.execute(
            "CREATE TABLE accepted.daily(ts_code VARCHAR, trade_date DATE)"
        )
        assert publish_derived_datasets(connection)
        row = connection.execute(
            "SELECT is_open, evidence_basis, confidence "
            "FROM derived.cn_equity_trade_calendar WHERE exchange='BSE'"
        ).fetchone()
    assert row == (None, 'unresolved', 'unresolved')


def test_audit_separates_missing_source_calendar_from_usable_derived_calendar(tmp_path):
    """场景：北交所原始日历缺失但当天有行情；预期：来源缺口失败，派生日历产品检查通过。"""
    database = tmp_path / 'reference.duckdb'
    with duckdb.connect(str(database)) as connection:
        connection.execute('CREATE SCHEMA accepted')
        connection.execute(
            "CREATE TABLE accepted.daily AS SELECT '920001.BJ' ts_code, "
            "DATE '2026-09-30' trade_date"
        )
        connection.execute('CREATE TABLE accepted.daily_basic AS SELECT * FROM accepted.daily')
        connection.execute('CREATE TABLE accepted.adj_factor AS SELECT * FROM accepted.daily')
        connection.execute(
            "CREATE TABLE accepted.stock_basic AS SELECT '920001.BJ' ts_code, "
            "DATE '2020-01-01' list_date, NULL::DATE delist_date"
        )
        connection.execute(
            "CREATE TABLE accepted.suspend_d(ts_code VARCHAR, trade_date DATE, "
            "suspend_type VARCHAR, suspend_timing VARCHAR)"
        )
        connection.execute(
            "CREATE TABLE accepted.trade_cal AS SELECT * FROM (VALUES "
            "('SSE', DATE '2026-09-30', '1'), ('SZSE', DATE '2026-09-30', '1')) "
            "t(exchange, cal_date, is_open)"
        )
        assert publish_derived_datasets(connection)
    checks = {item['name']: item for item in audit_reference(tmp_path)['checks']}
    assert checks['trade_calendar']['status'] == 'failed'
    assert checks['trade_calendar']['finding_count'] == 1
    assert checks['decision_trade_calendar']['status'] == 'passed'
    assert checks['decision_trade_calendar']['finding_count'] == 0
