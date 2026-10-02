"""跨表验收使用完全合成的本地数据库，不访问提供方。"""

import duckdb
import pytest

from tushare_entitlement_hub.reference_audit import audit_reference


@pytest.mark.parametrize('scenario', ['missing', 'matched', 'mismatch'])
def test_cross_dataset_audit_states(tmp_path, scenario):
    """场景：缺表、相互一致、因子日历名录缺项；预期：未检查、通过和失败分别明确记录。"""
    with duckdb.connect(str(tmp_path / 'reference.duckdb')) as connection:
        connection.execute('CREATE SCHEMA accepted')
        if scenario != 'missing':
            connection.execute("CREATE TABLE accepted.daily AS SELECT 'A.SH' ts_code, "
                               "DATE '2026-09-30' trade_date")
            connection.execute('CREATE TABLE accepted.daily_basic AS SELECT * FROM accepted.daily')
            connection.execute('CREATE TABLE accepted.adj_factor AS SELECT * FROM accepted.daily')
            connection.execute("CREATE TABLE accepted.stock_basic AS SELECT 'A.SH' ts_code, "
                               "DATE '2020-01-01' list_date, NULL::DATE delist_date")
            connection.execute("CREATE TABLE accepted.suspend_d(ts_code VARCHAR, trade_date DATE, "
                               "suspend_type VARCHAR, suspend_timing VARCHAR)")
            connection.execute("CREATE TABLE accepted.trade_cal AS SELECT 'SSE' exchange, "
                               "DATE '2026-09-30' cal_date, '1' is_open")
            if scenario == 'mismatch':
                connection.execute('DELETE FROM accepted.adj_factor')
                connection.execute('DELETE FROM accepted.stock_basic')
                connection.execute("UPDATE accepted.trade_cal SET exchange='SZSE'")
    report = audit_reference(tmp_path)
    assert report['status'] == {'missing': 'not_checked', 'matched': 'passed',
                                'mismatch': 'failed'}[scenario]
    assert not report['historical_universe_certified']
    assert (tmp_path / 'quality-audit.json').is_file()
    if scenario == 'mismatch':
        assert sum(item['finding_count'] for item in report['checks']) == 3


def test_listing_gaps_distinguish_full_day_suspension_and_resumption(tmp_path):
    """场景：名录中有缺行情证券，分全天停牌、日内停牌和当日复牌；预期：仅全天停牌可解释。"""
    from tushare_entitlement_hub.reference_audit import listing_coverage

    with duckdb.connect() as connection:
        connection.execute('CREATE SCHEMA accepted')
        connection.execute("CREATE TABLE accepted.daily AS SELECT 'A.SH' ts_code, "
                           "DATE '2026-09-30' trade_date")
        connection.execute("CREATE TABLE accepted.stock_basic AS SELECT name ts_code, "
                           "DATE '2020-01-01' list_date, NULL::DATE delist_date "
                           "FROM (VALUES ('A.SH'), ('B.SH'), ('C.SH'), ('D.SH')) t(name)")
        connection.execute("CREATE TABLE accepted.suspend_d AS SELECT ts_code, "
                           "DATE '2026-09-30' trade_date, suspend_type, suspend_timing "
                           "FROM (VALUES ('B.SH','S',NULL), ('C.SH','S','09:30-10:30'), "
                           "('D.SH','S',NULL), ('D.SH','R',NULL)) "
                           "t(ts_code,suspend_type,suspend_timing)")
        result = listing_coverage(connection)
    assert result['status'] == 'failed' and result['finding_count'] == 2
    assert result['suspension_explained'] == [['B.SH', '2026-09-30']]
    assert result['findings'] == [['C.SH', '2026-09-30'], ['D.SH', '2026-09-30']]
