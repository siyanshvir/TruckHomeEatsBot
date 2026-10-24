"""Offline matched-seed knob sweep for the final profile (R3).

    nice -n 15 python3 tools/research_sweep.py run --exp E1 E2 --seeds s1 s2 s3 --rivals 3 7 \\
        --styles mixed brutal moving follower --days 20 --workers 2 --out data/research-sweep.jsonl
    python3 tools/research_sweep.py report --out data/research-sweep.jsonl [--exp E1] [--base final]

Configs: 'final' (default knobs 1.0/0.5/0.10, calibrator+competition on), 'p0.9','p1.1' (price_aggr),
'b0','b1' (breadth), 'r0.03','r0.30' (risk), 'nocal', 'nocomp', 'baseline', 'unrestricted'.
Knobs are FIXED (controller stubbed). Every config sees identical Market(seed, rivals, days, style).
Sim is approximate; reputation resets per market. Offline only; never contacts a server.
"""
import argparse
import itertools
import json
from multiprocessing import Pool
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bot import make_brain
from controller import Knobs
from game import Rules
from sim import Market

FEATURES = {'fair_choice': True, 'walk_over': True, 'salvage': True}


def cfg(price=1.0, breadth=0.5, risk=0.10, cal=True, comp=True, policy='final'):
    return dict(policy=policy, knobs=(price, breadth, risk), cal=cal, comp=comp)


CONFIGS = {
    'final': cfg(), 'p0.9': cfg(price=0.9), 'p1.1': cfg(price=1.1),
    'b0': cfg(breadth=0.0), 'b1': cfg(breadth=1.0),
    'r0.03': cfg(risk=0.03), 'r0.30': cfg(risk=0.30),
    'nocal': cfg(cal=False), 'nocomp': cfg(comp=False),
    'baseline': cfg(policy='baseline'), 'unrestricted': cfg(policy='unrestricted'),
}
EXPERIMENTS = {
    'E1': ['final', 'r0.03', 'r0.30'], 'E2': ['final', 'p0.9', 'p1.1'], 'E3': ['final', 'b0', 'b1'],
    'E4': ['final', 'nocal'], 'E5': ['final', 'nocomp'], 'E6': ['final', 'baseline', 'unrestricted'],
}


class FixedController:
    def __init__(self, knobs):
        self.knobs = Knobs(*knobs)

    def choose(self, context=None, greedy=False):
        return self.knobs

    def update(self, *a, **k):
        pass

    def new_market(self, *a, **k):
        pass

    def to_dict(self):
        return {}


def build(name):
    c = CONFIGS[name]
    if c['policy'] == 'baseline':
        return make_brain(samples=16)
    if c['policy'] == 'unrestricted':
        return make_brain(small_crowds=False, samples=16)
    from final_mode import make_final_brain
    brain = make_final_brain(samples=16)
    pol = brain.final_policy
    pol.controller = FixedController(c['knobs'])
    if not c['cal']:
        pol.calibrator = None
    if not c['comp']:
        pol.competition = None
    return brain


def run_one(task):
    name, seed, rivals, style, days = task
    brain = build(name)
    market = Market(seed, rivals, days, style, features=dict(FEATURES))
    state = market.state()
    profits, waste, lat = [], 0, []
    for _ in range(days):
        t = time.perf_counter()
        order = brain.decide(state)
        lat.append(time.perf_counter()-t)
        Rules.read(state).validate(order)
        state = market.play(order)
        own = state['last_day']['you']
        profits.append(own['profit'])
        waste += own['meals']-own['sold']
    return dict(config=name, seed=seed, rivals=rivals, style=style, days=days,
                profit=round(state['you']['cash'], 2), worst_day=round(min(profits), 2),
                waste=waste, decision_mean=statistics.mean(lat))


def key(r):
    return (r['seed'], r['rivals'], r['style'], r['days'])


def load(path):
    rows = []
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run(args):
    names = []
    for e in args.exp:
        names += [n for n in EXPERIMENTS[e] if n not in names]
    done = {(r['config'],)+key(r) for r in load(args.out)}
    tasks = [(n, s, r, st, args.days) for s, r, st in itertools.product(args.seeds, args.rivals, args.styles)
             for n in names if (n, s, r, st, args.days) not in done]
    print('%d tasks' % len(tasks), flush=True)
    with Pool(min(args.workers, 2)) as pool, open(args.out, 'a') as fh:
        for res in pool.imap_unordered(run_one, tasks):
            fh.write(json.dumps(res)+'\n')
            fh.flush()


def report(args):
    rows = [r for r in load(args.out) if not args.days or r['days'] == args.days]
    by = {}
    for r in rows:
        by.setdefault(r['config'], {})[key(r)] = r
    base = by.get(args.base, {})
    names = [n for n in EXPERIMENTS.get(args.exp, by) if n in by] if args.exp else list(by)
    print('%-13s %4s %9s %8s %9s %8s %7s %6s | paired vs %s: n  mean  se  W/L' % (
        'config', 'n', 'mean', 'std', 'worstmkt', 'worstday', 'waste', 'sec', args.base))
    for n in names:
        g = by[n]
        p = [r['profit'] for r in g.values()]
        line = '%-13s %4d %9.1f %8.1f %9.1f %8.1f %7.1f %6.2f' % (
            n, len(p), statistics.mean(p), statistics.pstdev(p), min(p), min(r['worst_day'] for r in g.values()),
            statistics.mean(r['waste'] for r in g.values()), statistics.mean(r['decision_mean'] for r in g.values()))
        if n != args.base:
            d = [g[k]['profit']-base[k]['profit'] for k in g if k in base]
            if len(d) > 1:
                se = statistics.stdev(d)/len(d)**.5
                line += ' | %3d %+8.1f %6.1f %d/%d' % (len(d), statistics.mean(d), se,
                                                       sum(x > 0 for x in d), sum(x < 0 for x in d))
        print(line)
    if args.split:
        for field in ('style', 'rivals'):
            print('-- paired mean diff by %s' % field)
            for n in names:
                if n == args.base:
                    continue
                parts = []
                for v in sorted({r[field] for r in rows}, key=str):
                    d = [by[n][k]['profit']-base[k]['profit'] for k in by[n] if k in base and dict(zip(('seed', 'rivals', 'style'), k))[field] == v]
                    if d:
                        parts.append('%s=%+.0f(n%d)' % (v, statistics.mean(d), len(d)))
                print('%-13s %s' % (n, ' '.join(parts)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--exp', nargs='+', choices=sorted(EXPERIMENTS), required=True)
    r.add_argument('--seeds', nargs='+', default=['r3a', 'r3b', 'r3c'])
    r.add_argument('--rivals', type=int, nargs='+', default=[3, 7])
    r.add_argument('--styles', nargs='+', default=['mixed', 'brutal', 'moving', 'follower'])
    r.add_argument('--days', type=int, default=20)
    r.add_argument('--workers', type=int, default=2)
    r.add_argument('--out', default='data/research-sweep.jsonl')
    q = sub.add_parser('report')
    q.add_argument('--out', default='data/research-sweep.jsonl')
    q.add_argument('--exp', choices=sorted(EXPERIMENTS))
    q.add_argument('--base', default='final')
    q.add_argument('--days', type=int)
    q.add_argument('--split', action='store_true')
    args = ap.parse_args()
    (run if args.cmd == 'run' else report)(args)


if __name__ == '__main__':
    main()
