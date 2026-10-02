"""跨接口核对本地参考的数据覆盖；不以单表通过代替日历、证券范围和历史资格。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from .reference_storage import atomic_json


def audit_reference(destination: Path) -> dict[str, Any]:
    """核对A股日行情与每日指标、复权因子、证券名录及交易日历，并保存独立验收报告。

    输入含reference.duckdb的发布目录，输出逐检查结果。例：daily中有A但adj_factor无A，
    返回failed和missing记录；没有所需表则返回not_checked，绝不视为通过。
    核对仅覆盖已采日期及已观察证券名录，不证明历史全市场完整性或历史PIT资格。
    """
    checks: list[dict[str, Any]] = []
    with duckdb.connect(str(destination / 'reference.duckdb'), read_only=True) as connection:
        columns: dict[str, set[str]] = {}
        for table, column in connection.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema='accepted'"
        ).fetchall():
            columns.setdefault(table, set()).add(column)
        specifications = {
            'daily_basic_keys': ({'daily': {'ts_code', 'trade_date'},
                                  'daily_basic': {'ts_code', 'trade_date'}}, 'daily_basic'),
            'adj_factor_keys': ({'daily': {'ts_code', 'trade_date'},
                                 'adj_factor': {'ts_code', 'trade_date'}}, 'adj_factor'),
            'stock_membership': ({'daily': {'ts_code'}, 'stock_basic': {'ts_code'}}, None),
            'trade_calendar': ({'daily': {'ts_code', 'trade_date'},
                                'trade_cal': {'exchange', 'cal_date', 'is_open'}}, None),
        }
        specifications['observed_listing_coverage'] = ({
            'daily': {'ts_code', 'trade_date'},
            'stock_basic': {'ts_code', 'list_date', 'delist_date'},
            'suspend_d': {'ts_code', 'trade_date', 'suspend_type', 'suspend_timing'},
        }, None)
        for name, (requirements, comparison) in specifications.items():
            missing = {table: sorted(fields - columns.get(table, set()))
                       for table, fields in requirements.items()
                       if fields - columns.get(table, set())}
            if missing:
                checks.append({'name': name, 'status': 'not_checked',
                               'missing_columns': missing})
                continue
            if name == 'observed_listing_coverage':
                checks.append(listing_coverage(connection))
                continue
            if comparison:
                findings = connection.execute(
                    f'SELECT d.ts_code, d.trade_date FROM accepted.daily d '
                    f'ANTI JOIN accepted."{comparison}" r USING (ts_code, trade_date) '
                    'ORDER BY d.trade_date, d.ts_code'
                ).fetchall()
            elif name == 'stock_membership':
                findings = connection.execute(
                    'SELECT DISTINCT d.ts_code FROM accepted.daily d '
                    'ANTI JOIN accepted.stock_basic r USING (ts_code) ORDER BY d.ts_code'
                ).fetchall()
            else:
                findings = connection.execute(
                    "SELECT DISTINCT d.ts_code, d.trade_date FROM accepted.daily d "
                    "WHERE NOT EXISTS (SELECT 1 FROM accepted.trade_cal c "
                    "WHERE c.cal_date=d.trade_date AND CAST(c.is_open AS VARCHAR)='1' "
                    "AND c.exchange=CASE split_part(d.ts_code, '.', 2) "
                    "WHEN 'SH' THEN 'SSE' WHEN 'SZ' THEN 'SZSE' WHEN 'BJ' THEN 'BSE' END) "
                    "ORDER BY d.trade_date, d.ts_code"
                ).fetchall()
            checks.append({'name': name, 'status': 'failed' if findings else 'passed',
                           'finding_count': len(findings),
                           'findings': [[str(value) for value in row] for row in findings]})
    result = {
        'scope': 'observed_a_share_daily_cross_dataset_checks', 'checks': checks,
        'status': 'failed' if any(item['status'] == 'failed' for item in checks)
        else 'not_checked' if any(item['status'] == 'not_checked' for item in checks) else 'passed',
        'historical_universe_certified': False, 'historical_pit_certified': False,
        'limitations': [
            '证券范围按当前名录的上市与退市日期推导，只核对daily已出现的日期',
            '停牌仅解释当日S类型、未注明日内时段且当日无复牌的缺行情；未证明历史事件齐全',
            '交易日历缺对应交易所时报告缺口，不借用另一交易所日历',
            '其余市场及接口尚未执行跨表完整性验收',
        ],
    }
    atomic_json(destination / 'quality-audit.json', result)
    return result


def listing_coverage(connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """按观察到的上市退市日期核对每日应有证券，并分开记录停牌可解释和无法解释的缺项。

    输入已验证所需列的本地连接，输出检查结果；例如一只证券全天停牌无行情，
    保存到suspension_explained，而无停牌依据的证券保存到findings并令检查失败。
    只检查daily已有日期；名录和停牌记录的历史完整性不能由本检查证明。
    """
    rows = connection.execute(
        "WITH missing AS (SELECT s.ts_code, d.trade_date FROM accepted.stock_basic s "
        "CROSS JOIN (SELECT DISTINCT trade_date FROM accepted.daily) d "
        "WHERE s.list_date <= d.trade_date "
        "AND (s.delist_date IS NULL OR s.delist_date > d.trade_date) "
        "AND NOT EXISTS (SELECT 1 FROM accepted.daily q "
        "WHERE q.ts_code=s.ts_code AND q.trade_date=d.trade_date)) "
        "SELECT m.ts_code, m.trade_date, "
        "EXISTS (SELECT 1 FROM accepted.suspend_d s WHERE s.ts_code=m.ts_code "
        "AND s.trade_date=m.trade_date AND s.suspend_type='S' "
        "AND nullif(trim(s.suspend_timing), '') IS NULL) "
        "AND NOT EXISTS (SELECT 1 FROM accepted.suspend_d s WHERE s.ts_code=m.ts_code "
        "AND s.trade_date=m.trade_date AND s.suspend_type='R') AS explained "
        "FROM missing m ORDER BY m.trade_date, m.ts_code"
    ).fetchall()
    unexplained = [[str(code), str(day)] for code, day, explained in rows if not explained]
    explained_rows = [[str(code), str(day)] for code, day, explained in rows if explained]
    return {'name': 'observed_listing_coverage', 'status': 'failed' if unexplained else 'passed',
            'finding_count': len(unexplained), 'findings': unexplained,
            'suspension_explained': explained_rows}
