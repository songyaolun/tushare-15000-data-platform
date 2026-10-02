"""从已发布且可追溯的来源表生成产品可用派生参考，不改写提供方原始事实。"""

from __future__ import annotations

import duckdb


def publish_derived_datasets(connection: duckdb.DuckDBPyConnection) -> bool:
    """发布带证据等级的中国权益交易日历，来源表不足时仅创建derived schema。

    输入已加载accepted数据集的DuckDB连接，成功创建日历返回True，缺少trade_cal或daily
    返回False。例：上深同为开市且当天存在.BJ行情时，BSE行标为实际市场活动证据；
    上深一致但未采北交所行情时只标为日历对齐推导，二者不一致时is_open保持NULL。
    """
    connection.execute("CREATE SCHEMA IF NOT EXISTS derived")
    available = {
        row[0]
        for row in connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='accepted'"
        ).fetchall()
    }
    if not {"daily", "trade_cal"} <= available:
        return False
    connection.execute("DROP TABLE IF EXISTS derived.cn_equity_trade_calendar")
    connection.execute(
        "CREATE TABLE derived.cn_equity_trade_calendar AS "
        "WITH official AS ("
        "  SELECT exchange, cal_date, CAST(CAST(is_open AS VARCHAR)='1' AS BOOLEAN) is_open "
        "  FROM accepted.trade_cal WHERE exchange IN ('SSE','SZSE')"
        "), official_signals AS ("
        "  SELECT cal_date, "
        "    max(is_open::INTEGER) FILTER (exchange='SSE')::BOOLEAN sse_open, "
        "    max(is_open::INTEGER) FILTER (exchange='SZSE')::BOOLEAN szse_open "
        "  FROM official GROUP BY cal_date"
        "), bse_activity AS ("
        "  SELECT trade_date cal_date, count(DISTINCT ts_code)::BIGINT observed_security_count "
        "  FROM accepted.daily WHERE ts_code LIKE '%.BJ' GROUP BY trade_date"
        "), calendar_dates AS ("
        "  SELECT cal_date FROM official_signals UNION SELECT cal_date FROM bse_activity"
        "), bse AS ("
        "  SELECT 'BSE'::VARCHAR exchange, dates.cal_date, "
        "    CASE WHEN coalesce(activity.observed_security_count,0)>0 THEN true "
        "         WHEN signals.sse_open=signals.szse_open THEN signals.sse_open "
        "         ELSE NULL END::BOOLEAN is_open, "
        "    CASE WHEN coalesce(activity.observed_security_count,0)>0 "
        "           THEN 'observed_bse_market_activity' "
        "         WHEN signals.sse_open=signals.szse_open "
        "           THEN 'aligned_sse_szse_calendar' ELSE 'unresolved' END evidence_basis, "
        "    ['trade_cal','daily']::VARCHAR[] source_apis, "
        "    coalesce(activity.observed_security_count,0)::BIGINT observed_security_count, "
        "    CASE WHEN coalesce(activity.observed_security_count,0)>0 THEN 'high' "
        "         WHEN signals.sse_open=signals.szse_open THEN 'medium' "
        "         ELSE 'unresolved' END confidence, false historical_pit_certified "
        "  FROM calendar_dates dates LEFT JOIN official_signals signals USING (cal_date) "
        "  LEFT JOIN bse_activity activity USING (cal_date)"
        ") "
        "SELECT exchange, cal_date, is_open, 'official_exchange_calendar' evidence_basis, "
        "  ['trade_cal']::VARCHAR[] source_apis, 0::BIGINT observed_security_count, "
        "  'high' confidence, false historical_pit_certified FROM official "
        "UNION ALL SELECT * FROM bse"
    )
    return True
