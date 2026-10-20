"""ETF正式数据门禁：合成来源离线验证，不需要凭据或网络。"""

import hashlib
import json

import pytest

from tushare_entitlement_hub.etf_research import data


def write_source(root, api, accepted, quarantine=(), complete=True):
    """建立带真实校验和的合成来源；例：一行记录生成可验证manifest与JSONL。"""
    folder = root / 'datasets' / api
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in [('accepted', accepted), ('quarantine', quarantine)]:
        path = folder / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        files[name] = {'jsonl': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'rows': len(rows)}
    manifest = {'collection_status': 'complete_provider_response' if complete else 'failed',
                'parameters': {'ts_code': '510300.SH'}, 'observed_at': '2026-10-18T00:00:00Z',
                'files': files}
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    return manifest


def cash_event(**changes):
    """生成已实施现金派息；默认每份.1元，登记20210104、除息20210105、支付20210106。"""
    return {'ts_code': '510300.SH', 'ann_date': '20210101', 'imp_anndate': '20210101',
            'record_date': '20210104', 'ex_date': '20210105', 'pay_date': '20210106',
            'div_cash': .1, 'div_proc': '实施', **changes}


def test_project_cash_relevant_duplicate_records(tmp_path):
    """场景：隔离仅因总份额冲突且现金字段一致；预期保留记录供严格现金投影去重。"""
    rows = [cash_event(base_unit=10, issues=['conflicting_business_key']),
            cash_event(base_unit=20, issues=['conflicting_business_key'])]
    write_source(tmp_path, 'fund_div', [], rows)
    records, evidence = data.source_rows(tmp_path, 'fund_div')
    assert len(records) == 2
    assert evidence['cash_projection_quarantine_rows'] == 2
    assert len(data.dividend_events(records, '20210101', '20211231')) == 1


def source_fixture():
    """生成三个交易日实际价格与复权来源；例：返回的fund_daily含唯一日期和正常价格。"""
    days = ['20210104', '20210105', '20210106']
    return {
        'fund_daily': [dict(ts_code='510300.SH', trade_date=day, open=1, high=1,
                            low=1, close=1, vol=100) for day in days],
        'fund_adj': [dict(ts_code='510300.SH', trade_date=day, adj_factor=1) for day in days],
        'etf_limit': [dict(ts_code='510300.SH', trade_date=day, up_limit=2, down_limit=.5)
                      for day in days],
        'fund_div': [],
        'trade_cal': [dict(cal_date=day, is_open=1) for day in days],
    }


def test_complete_data_and_dates(tmp_path):
    """场景：五个接口正式清单完整且校验通过；预期载入价格并记录每个来源的校验和。"""
    rows = source_fixture()
    for api, values in rows.items():
        write_source(tmp_path, api, values)
    calendar, bars, events, audit = data.prepare_data(rows, '20210101', '20211231')
    assert len(calendar) == len(bars) == 3
    assert not events
    assert audit['security'] == '510300.SH'
    assert data.date_key('2021-01-04') == '20210104'
    assert data.date_key(None) == ''
    with pytest.raises(ValueError, match='source_calendar'):
        data.load_data(tmp_path)


@pytest.mark.parametrize(
    'failure', ['incomplete', 'integrity', 'missing_file', 'quarantine', 'issue']
)
def test_source_rejects_incomplete_or_unprojectable(tmp_path, monkeypatch, failure):
    """场景：来源不完整、校验损坏或非重复冲突隔离；预期阻止回放，不静默丢弃。"""
    api = 'fund_daily' if failure == 'quarantine' else 'fund_div'
    extra = [cash_event(issues=['invalid'])] if failure in {'quarantine', 'issue'} else []
    manifest = write_source(tmp_path, api, [], extra, complete=failure != 'incomplete')
    if failure == 'integrity':
        monkeypatch.setattr(data, 'verified_files', lambda value: False)
    if failure == 'missing_file':
        manifest['files']['accepted'] = {}
        (tmp_path / 'datasets' / api / 'manifest.json').write_text(json.dumps(manifest))
        monkeypatch.setattr(data, 'verified_files', lambda value: True)
    with pytest.raises(ValueError):
        data.source_rows(tmp_path, api)


@pytest.mark.parametrize('changes', [
    {'record_date': None}, {'pay_date': '20210104'}, {'div_cash': -1},
    {'div_cash': float('nan')},
])
def test_dividend_required_fields(changes):
    """场景：权益日期缺失倒置或金额非法；预期拒绝现金派息记录。"""
    with pytest.raises(ValueError):
        data.dividend_events([cash_event(**changes)], '20210101', '20211231')


@pytest.mark.parametrize('changes', [{'div_cash': .2}, {'ann_date': '20201231'},
                                    {'pay_date': '20210107'}])
def test_cash_projection_conflict_not_silently_selected(changes):
    """场景：重复事件的现金、公告或支付字段冲突；预期拒绝，不任取一个版本。"""
    with pytest.raises(ValueError, match='conflicting'):
        data.dividend_events([cash_event(), cash_event(**changes)], '20210101', '20211231')
    assert data.dividend_events([cash_event(div_proc='预案')], '20210101', '20211231') == []


@pytest.mark.parametrize('kind', ['code', 'duplicate', 'calendar_duplicate', 'empty_calendar',
                                 'missing', 'price', 'volume', 'range', 'limit', 'factor'])
def test_history_must_be_real_complete_and_valid(kind):
    """场景：错误代码、日历缺口、非法价格成交量或无解释复权变化；预期关闭回放门禁。"""
    rows = source_fixture()
    if kind == 'code':
        rows['fund_daily'][0]['ts_code'] = '000300.SH'
    elif kind == 'duplicate':
        rows['fund_daily'].append(rows['fund_daily'][0])
    elif kind == 'calendar_duplicate':
        rows['trade_cal'].append(rows['trade_cal'][0])
    elif kind == 'empty_calendar':
        rows['trade_cal'] = []
    elif kind == 'missing':
        rows['fund_daily'].pop()
    elif kind == 'price':
        rows['fund_daily'][0]['open'] = 0
    elif kind == 'volume':
        rows['fund_daily'][0]['vol'] = float('nan')
    elif kind == 'range':
        rows['fund_daily'][0]['high'] = .8
    elif kind == 'limit':
        rows['etf_limit'][0]['up_limit'] = .9
    elif kind == 'factor':
        rows['fund_adj'][1]['adj_factor'] = 2
    with pytest.raises(ValueError):
        data.prepare_data(rows, '20210101', '20211231')


def test_cash_factor_reconciliation():
    """场景：每份除息.1对应因子1/.9；预期核对通过，无法解释的比例或缺登记日拒绝。"""
    rows = source_fixture()
    rows['fund_div'] = [cash_event()]
    for row in rows['fund_adj'][1:]:
        row['adj_factor'] = 1 / .9
    assert len(data.prepare_data(rows, '20210101', '20211231')[2]) == 1
    rows['fund_div'][0]['record_date'] = '20210103'
    with pytest.raises(ValueError, match='calendar'):
        data.prepare_data(rows, '20210101', '20211231')
    rows['fund_div'][0] = cash_event(div_cash=2)
    with pytest.raises(ValueError, match='cash_adjustment_invalid'):
        data.prepare_data(rows, '20210101', '20211231')
    rows['fund_div'][0] = cash_event(div_cash=.2)
    with pytest.raises(ValueError, match='cash_adjustment_mismatch'):
        data.prepare_data(rows, '20210101', '20211231')


def test_load_data_passes_validated_sources(tmp_path, monkeypatch):
    """场景：全部来源通过；预期load_data返回原始审计并增加五个来源证据。"""
    for api, values in source_fixture().items():
        write_source(tmp_path, api, values)
    original = data.prepare_data
    monkeypatch.setattr(data, 'prepare_data', lambda rows: original(rows, '20210101', '20211231'))
    assert len(data.load_data(tmp_path)[3]['sources']) == 5
