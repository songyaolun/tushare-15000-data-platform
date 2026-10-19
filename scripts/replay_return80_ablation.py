"""Replay frozen scores with explicit capital and account universe; never retrain."""

import argparse
import hashlib
import json
import runpy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from tushare_entitlement_hub.regime_research.attribution import evaluate_portfolio
from tushare_entitlement_hub.regime_research.liquidity import LiquidityRule
from tushare_entitlement_hub.strategy_discovery.audit import audit_result
from tushare_entitlement_hub.strategy_discovery.candidate_allocation import (
    simulate_candidate_allocation,
)
from tushare_entitlement_hub.strategy_discovery.experiment_scope import account_scope_mask
from tushare_entitlement_hub.strategy_discovery.replacement import (
    ReplacementRule,
    make_replacement_provider,
)
from tushare_entitlement_hub.strategy_discovery.replay_parameters import ReplayParameters
from tushare_entitlement_hub.strategy_discovery.runner import (
    additional_metrics,
    configuration,
    save_result,
)


def main():
    """Run normal/double-cost accounts; --output fresh-dir creates immutable CSV evidence.

    --scores defaults to the original model. --known-mask optionally restricts both
    baseline and experiment to an identical candidate availability matrix; --start
    sets the same fresh-account start date in both. Never infer absent values as zero.
    Example: --capital 1000000 --scope original replays the original candidate pool;
    defaults remain --capital 120000 --scope mainboard for earlier experiments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', type=Path)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--known-mask', type=Path)
    parser.add_argument('--start', default='20211018')
    parser.add_argument('--name', default='baseline')
    parser.add_argument('--capital', type=float, default=120000)
    parser.add_argument('--scope', choices=('mainboard', 'original'), default='mainboard')
    parser.add_argument('--take-profit', type=float, default=.045)
    parser.add_argument('--stop-loss', type=float, default=.08)
    parser.add_argument('--maximum-holding', type=int, default=30)
    parser.add_argument('--position-weight', type=float, default=.05)
    parser.add_argument('--replacement-sessions', type=int, default=10)
    parser.add_argument('--capture-execution-evidence', action='store_true')
    parser.add_argument('--maximum-daily-participation', type=float)
    parser.add_argument('--liquidity-lookback', type=int, default=20)
    args = parser.parse_args()
    liquidity_rule = LiquidityRule(args.maximum_daily_participation, args.liquidity_lookback) \
        if args.maximum_daily_participation is not None else None
    parameters = ReplayParameters(
        args.take_profit, args.stop_loss, args.maximum_holding,
        args.position_weight, args.replacement_sessions,
    )
    if not np.isfinite(args.capital) or args.capital <= 0:
        raise ValueError('invalid_initial_capital')
    cache = args.cache_root
    scores_path = args.scores or cache / (
        'stockops-return80-model-20261019/scores/hgb_leaves31_rounds1600.npy'
    )
    if args.output.exists():
        raise ValueError('fresh_output_required')
    context_path = cache / 'stockops-candidate-allocation-v2-20261019/run.py'
    cap_path = cache / 'stockops-sector-cap-20261019/run.py'
    context = runpy.run_path(str(context_path))['context']()
    panel, _, account_policy, base, events, original_states, factors = context
    sector_mask = runpy.run_path(str(cap_path))['sector_capped_mask']
    scores = np.load(scores_path, mmap_mode='r')
    if scores.shape != base.entry_mask.shape or np.isinf(scores).any():
        raise ValueError('invalid_score_axis')
    scope = account_scope_mask(panel.codes, args.scope)
    eligible = base.entry_mask & scope[None, :]
    if args.known_mask:
        known = np.load(args.known_mask, mmap_mode='r')
        if known.shape != eligible.shape or known.dtype != bool:
            raise ValueError('invalid_known_mask')
        eligible &= known
    if not np.isfinite(scores[eligible]).all():
        raise ValueError('missing_scores_on_eligible_candidates')
    selected = sector_mask(eligible, scores, panel.values['sector_index'], 1)
    states = (np.isfinite(panel.market['breadth']) & (panel.market['breadth'] >= .4)).astype(
        np.int8
    )
    definition = replace(
        base, entry_mask=selected, scores=scores, maximum_entry_gap=.05,
        policy=replace(base.policy, take_profit=parameters.take_profit,
                       stop_loss=parameters.stop_loss,
                       maximum_holding=parameters.maximum_holding, take_profit_enabled=True),
    )
    rule = ReplacementRule(parameters.replacement_sessions, .01, 'momentum')
    args.output.mkdir(parents=True)
    manifest = {
        'protocol': 'return80-capital-scope-ablation/v2',
        'capital': args.capital, 'start': args.start, 'end': panel.calendar[-1],
        'scope': 'scope-v1-c0-s0' if args.scope == 'mainboard' else 'original-frozen-entry-mask',
        'market_features': 'unchanged full market',
        'score_file': str(scores_path),
        'score_sha256': hashlib.sha256(scores_path.read_bytes()).hexdigest(),
        'known_mask': str(args.known_mask) if args.known_mask else None,
        'known_mask_sha256': hashlib.sha256(args.known_mask.read_bytes()).hexdigest()
        if args.known_mask else None,
        'context_sha256': hashlib.sha256(context_path.read_bytes()).hexdigest(),
        'sector_selection_sha256': hashlib.sha256(cap_path.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope_securities': int(scope.sum()), 'mature_fit_scope': 'original full market',
        'sample': 'full-sample fit and selection; not out-of-sample validation',
        'replacement': asdict(rule), 'cost_multipliers': [1, 2],
        'execution_parameters': asdict(parameters),
        'capture_execution_evidence': args.capture_execution_evidence,
        'liquidity_rule': asdict(liquidity_rule) if liquidity_rule is not None else None,
    }
    (args.output / 'protocol.json').write_text(json.dumps(manifest, indent=2))
    reports = []
    for multiplier in (1, 2):
        name = f'{args.name}_cost{multiplier}'
        policy = replace(
            account_policy, allocations=(.15, 1., 1., 1., 1.),
            position_weight=parameters.position_weight,
            sector_weight=.4, fee_multiplier=multiplier, slippage_rate=.0005 * multiplier,
        )
        print(json.dumps({'start': name}), flush=True)
        result = simulate_candidate_allocation(
            panel, states, events, replace(definition, name=name), policy,
            mode='exit_reserve', capital=args.capital, start=args.start,
            replacement_provider=make_replacement_provider(rule),
            capture_execution_evidence=args.capture_execution_evidence,
            liquidity_rule=liquidity_rule,
        )
        selections = result.pop('candidate_selections')
        for selection in selections:
            if not scope[selection['selected']].all():
                raise ValueError('selection_outside_account_scope')
        for row in result['daily']:
            row['stage'] = int(original_states[row['day']])
        metrics = evaluate_portfolio(result, panel, factors) | additional_metrics(result)
        metrics.update({
            'mean_exposure': float(np.mean([row['exposure'] for row in result['daily']])),
            'replacement_exits': sum(row['reason'] == 'replacement' for row in result['trades']),
            'turnover': result['turnover'], 'total_fees': result['fees'],
            'final_assets': result['daily'][-1]['nav'],
            'sell_fills': len(result['trades']),
            'fully_exited_positions': len(result['entry_days']) - result['open_positions'],
        })
        directory = args.output / name
        directory.mkdir()
        save_result(directory, result, {
            'strategy': configuration(definition), 'account_policy': asdict(policy),
            'capital': args.capital, 'replacement_rule': asdict(rule), 'protocol': manifest,
        }, metrics)
        (directory / 'selections.json').write_text(json.dumps(selections, separators=(',', ':')))
        audited = audit_result(directory, initial_capital=args.capital)
        (directory / 'audit.json').write_text(json.dumps(audited, indent=2))
        reports.append({'name': name, 'cost': multiplier, **metrics})
        (args.output / 'results.json').write_text(json.dumps(reports, indent=2))
        print(json.dumps({'done': name, **{
            key: metrics[key] for key in (
                'annualized_return', 'maximum_drawdown', 'win_rate', 'closed_trades'
            )
        }}), flush=True)


if __name__ == '__main__':
    main()
