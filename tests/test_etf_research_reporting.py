"""ETF导出契约离线测试；真实行情不进入fixture。"""

import json

import pytest
from test_etf_research_engine import configuration, history

from tushare_entitlement_hub.etf_research import reporting


def test_report_exports_daily_orders_annual_and_rejects_overwrite(tmp_path, monkeypatch):
    """场景：三账户合成数据回放；预期各目录有逐日逐笔年度文件且禁止覆写。"""
    calendar, bars = history()
    monkeypatch.setattr(reporting, 'load_data', lambda root: (calendar, bars, [], {'valid': True}))
    monkeypatch.setattr(reporting, 'Configuration', configuration)
    calendar_file = tmp_path / 'calendar.json'
    calendar_file.write_text(json.dumps(calendar))
    output = tmp_path / 'results'
    summary = reporting.run_research(tmp_path, output, calendar_file)
    assert len(summary) == 3
    assert len(json.loads((output / 'summary.json').read_text())) == 3
    assert (output / 'buy_hold' / 'annual.csv').read_text().startswith('year,')
    assert (output / 'buy_hold' / 'dividends.csv').read_text() == ''
    assert (output / 'buy_hold' / 'daily.csv').exists()
    with pytest.raises(ValueError, match='already_exists'):
        reporting.run_research(tmp_path, output, calendar_file)


def test_independent_calendar_detects_shared_gaps(tmp_path):
    """场景：新采集接口共同漏一天；预期独立旧日历比对失败，固定目标日数也不可少。"""
    calendar, _ = history()
    path = tmp_path / 'calendar.json'
    path.write_text(json.dumps(calendar))
    with pytest.raises(ValueError, match='independent_calendar'):
        reporting.verify_calendar(calendar[:-1], path, configuration())
    path.write_text(json.dumps(['20211018', '20261016']))
    with pytest.raises(ValueError, match='session_count'):
        reporting.verify_calendar(['20211018', '20261016'], path, reporting.Configuration())
