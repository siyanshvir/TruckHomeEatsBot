"""Stress the policy in the harshest simulated fields and compare steering options.

    python3 tools/brutal.py                      # default sweep, 8 processes
Offline numbers are a stress signal only; validate in server practice and the live game.
"""
import statistics
import sys
import time
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bot import make_brain
from game import Rules
from sim import Market
from strategy import Brain

VARIANTS = {
    'baseline':              {},
    'cap 10':                {'price_cap': 10.0},
    'cap 10.5':              {'price_cap': 10.5},
    'quiet 40':              {'quiet_bonus': 40.},
    'quiet 100':             {'quiet_bonus': 100.},
    'cap 10 + quiet 40':     {'price_cap': 10.0, 'quiet_bonus': 40.},
    'cap 10 + quiet 100':    {'price_cap': 10.0, 'quiet_bonus': 100.},
    'cap 10 + quiet 100 + risk .5': {'price_cap': 10.0, 'quiet_bonus': 100., 'risk': .5},
}
FIELDS = (('brutal', 8), ('brutal', 12), ('undercut', 8), ('moving', 10))


def one(job):
    name, seed, style, rivals = job
    base = make_brain()
    brain = Brain(base.learner, seed=seed, **VARIANTS[name])
    market = Market(seed, rivals, 30, style)
    state = market.state()
    zero = 0
    for _ in range(30):
        order = brain.decide(state)
        state = market.play(order)
        own = state['last_day']['you']
        zero += own['sold'] == 0 and own['meals'] > 0
    others = [v for k, v in market.cash.items() if k != 'us']
    return name, style, rivals, market.cash['us'], max(others), zero


if __name__ == '__main__':
    seeds = ['b%d' % i for i in range(6)]
    jobs = [(n, sd, st, rv) for n in VARIANTS for sd in seeds for st, rv in FIELDS]
    t = time.time()
    with Pool(8) as pool:
        out = pool.map(one, jobs)
    print('%d markets in %.0fs' % (len(jobs), time.time() - t))
    for style, rv in FIELDS:
        print('\n== %s field, %d rivals' % (style, rv))
        for name in VARIANTS:
            r = [o for o in out if o[0] == name and o[1] == style and o[2] == rv]
            m = [x[3] for x in r]
            print('  %-30s avg %7.0f  min %7.0f  beats best rival %d/%d  zero-sale days %.1f' % (
                name, statistics.mean(m), min(m), sum(x[3] > x[4] for x in r), len(r), statistics.mean(x[5] for x in r)))
    print('\n== ALL FIELDS')
    for name in VARIANTS:
        m = [o[3] for o in out if o[0] == name]
        print('  %-30s avg %7.0f  min %7.0f' % (name, statistics.mean(m), min(m)))
