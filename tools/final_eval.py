"""Matched-seed offline comparison of the baseline and final policies.

    python3 tools/final_eval.py --seeds a b --rivals 3 7 --days 30 \\
        --styles mixed brutal moving follower --policies baseline final \\
        --features fair_choice walk_over salvage --markets 8 --output data/final-eval.json

Every policy gets the same simulated market (seed, rivals, style, flags) with a fresh learner.
Sim reputation resets with each market, like the real game's per-market reset, so no policy
carries reputation across markets here; cross-market learning is NOT exercised by this runner.
Offline numbers are an approximate stress signal; confirm in private practice.
"""
import argparse
import itertools
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bot import atomic_json, make_brain
from game import Rules
from sim import Market

POLICIES = ('baseline', 'unrestricted', 'final')


def make_policy(name, samples=None):
    """Return a fresh brain, or None (with a notice) when the policy is unavailable."""
    extra = {} if samples is None else {'samples': samples}
    if name == 'baseline':
        return make_brain(**extra)
    if name == 'unrestricted':
        return make_brain(small_crowds=False, **extra)
    if name == 'final':
        try:
            from final_mode import make_final_brain  # added by the lead; imported lazily
            return make_final_brain()
        except Exception as error:
            print('NOTICE: policy "final" skipped: %s: %s' % (type(error).__name__, error), flush=True)
            return None
    raise ValueError('unknown policy %r' % name)


def run_market(brain, seed, rivals, days, style, features):
    market = Market(seed, rivals, days, style, features=features)
    state = market.state()
    profits, waste, zero, latencies = [], 0, 0, []
    for _ in range(days):
        started = time.perf_counter()
        order = brain.decide(state)
        latencies.append(time.perf_counter()-started)
        Rules.read(state).validate(order)
        state = market.play(order)
        own = state['last_day']['you']
        profits.append(own['profit'])
        waste += own['meals']-own['sold']
        zero += own['sold'] == 0 and own['meals'] > 0
    return {'profit': round(state['you']['cash'], 2), 'waste': waste, 'zero_sale_days': zero,
            'worst_day': round(min(profits), 2), 'decision_max_seconds': max(latencies),
            'decision_mean_seconds': statistics.mean(latencies)}


def summarize(results, policies):
    summary = {}
    for name in policies:
        group = [r for r in results if r['policy'] == name]
        if not group:
            continue
        worst = min(group, key=lambda r: r['profit'])
        summary[name] = {'markets': len(group),
                         'average_profit': statistics.mean(r['profit'] for r in group),
                         'average_waste': statistics.mean(r['waste'] for r in group),
                         'zero_sale_days': sum(r['zero_sale_days'] for r in group),
                         'worst_day': min(r['worst_day'] for r in group),
                         'worst_market': {k: worst[k] for k in ('seed', 'rivals', 'style', 'profit')},
                         'decision_max_seconds': max(r['decision_max_seconds'] for r in group),
                         'decision_mean_seconds': statistics.mean(r['decision_mean_seconds'] for r in group)}
    return summary


def regressions(results, candidate='final', base='baseline'):
    """Per-market profit difference candidate - base; negatives are regressions."""
    index = {(r['policy'], r['seed'], r['rivals'], r['style']): r for r in results}
    rows = []
    for (policy, seed, rivals, style), r in sorted(index.items(), key=lambda kv: str(kv[0])):
        other = index.get((base, seed, rivals, style))
        if policy == candidate and other:
            rows.append({'seed': seed, 'rivals': rivals, 'style': style,
                         'profit_delta': round(r['profit']-other['profit'], 2),
                         'waste_delta': r['waste']-other['waste'],
                         'zero_sale_delta': r['zero_sale_days']-other['zero_sale_days'],
                         'worst_day_delta': round(r['worst_day']-other['worst_day'], 2)})
    return {'candidate': candidate, 'base': base, 'markets': rows,
            'regressed': [x for x in rows if x['profit_delta'] < 0]}


def evaluate(seeds, rivals, days, styles, policies, features=(), markets=None, samples=None, log=None):
    flags = {name: True for name in features}
    combos = list(itertools.product(seeds, rivals, styles))
    if markets:
        combos = combos[:markets]
    available = [p for p in policies if p != 'final' or make_policy('final') is not None]
    skipped = [p for p in policies if p not in available]
    results = []
    for seed, count, style in combos:
        for name in available:
            brain = make_policy(name, samples)  # fresh learner per policy and market
            result = dict(policy=name, seed=seed, rivals=count, style=style, days=days,
                          **run_market(brain, seed, count, days, style, flags))
            results.append(result)
            if log:
                log(json.dumps(result))
    report = {'source': 'approximate-offline', 'days': days, 'features': sorted(flags),
              'note': 'Sim reputation resets every market; cross-market learning is not exercised.',
              'policies': available, 'skipped_policies': skipped, 'results': results,
              'summary': summarize(results, available)}
    if 'final' in available and 'baseline' in available:
        report['regressions'] = regressions(results)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seeds', nargs='+', default=['unseen-a', 'unseen-b'])
    parser.add_argument('--rivals', type=int, nargs='+', default=[3, 7])
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--styles', nargs='+', default=['mixed', 'brutal', 'moving', 'follower'])
    parser.add_argument('--policies', nargs='+', default=['baseline', 'final'], choices=POLICIES)
    parser.add_argument('--features', nargs='*', choices=['fair_choice', 'walk_over', 'salvage'], default=[])
    parser.add_argument('--markets', type=int, default=None, help='Limit to the first N seed/rivals/style markets')
    parser.add_argument('--samples', type=int, default=None, help='Planner samples for baseline/unrestricted')
    parser.add_argument('--output', default='data/final-eval.json')
    args = parser.parse_args(argv)
    report = evaluate(args.seeds, args.rivals, args.days, args.styles, args.policies, args.features,
                      args.markets, args.samples, log=lambda line: print(line, flush=True))
    atomic_json(args.output, report)
    for name, s in report['summary'].items():
        print('%s: average $%.2f, worst market $%.2f (%s), waste %.1f, zero-sale days %d, worst day $%.2f, '
              'decision max %.3fs mean %.3fs' % (name, s['average_profit'], s['worst_market']['profit'],
                                                 s['worst_market']['style'], s['average_waste'],
                                                 s['zero_sale_days'], s['worst_day'],
                                                 s['decision_max_seconds'], s['decision_mean_seconds']))
    for skipped in report['skipped_policies']:
        print('skipped policy: %s' % skipped)
    for row in report.get('regressions', {}).get('regressed', []):
        print('REGRESSION %s/%s/%s: profit %+.2f' % (row['seed'], row['rivals'], row['style'], row['profit_delta']))
    print('note: %s' % report['note'])
    return report


if __name__ == '__main__':
    main()
