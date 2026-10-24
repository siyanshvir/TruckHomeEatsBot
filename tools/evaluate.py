"""Matched-seed comparisons. Each policy/market owns a fresh learner and rival history."""
import argparse
import json
import random
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bot import Client, atomic_json, load_env, make_brain
from game import Rules
from learning import Learner
from strategy import Brain
from sim import Market


class Cautious:
    """Simple baseline: premium at $9.25, forecast minus yesterday's rival stock."""
    def decide(self, state):
        rules = Rules.read(state)
        forecast = state['forecast']['expected_customers']
        last = (state.get('last_day') or {}).get('spots') or {}
        def free(spot):
            return max(0, forecast.get(spot, 0)-sum(t['meals'] for t in last.get(spot, {}).get('trucks', [])
                                                  if t['id'] != state['you']['id']))
        spot = max(rules.spots, key=free)
        return {'spot': spot, 'quality': 'premium', 'price': 9.25,
                'meals': min(rules.capacity('premium'), max(0, int(free(spot)*.7)))}


def policy(name):
    if name == 'learned': return make_brain()
    if name == 'untrained': return Brain(Learner())
    if name == 'cautious': return Cautious()
    raise ValueError(name)


def request(client, method, path, body=None):
    for attempt in range(6):
        try:
            status, result = client.call(method, path, body)
            if status in (200, 201): return result
            if status != 429 and status < 500:
                raise RuntimeError('Practice HTTP %d' % status)
        except OSError:
            pass
        time.sleep(min(5, .5*2**attempt))
    raise RuntimeError('Practice request failed repeatedly')


def run(name, seed, rivals, days, style, client=None):
    brain = policy(name)  # No shared mutable global, even between sequential markets.
    market = None if client else Market(seed, rivals, days, style)
    if client:
        created = request(client, 'POST', '/api/v1/practice', {'seed': seed, 'rivals': rivals, 'days': days})
        base = created['base']
        state = request(client, 'GET', base+'/state')
    else:
        state = market.state()
    latencies, profits, waste, zero = [], [], 0, 0
    for _ in range(days):
        if state['phase'] != 'night': raise RuntimeError('Practice ended before expected day count')
        started = time.perf_counter()
        order = brain.decide(state)
        latencies.append(time.perf_counter()-started)
        Rules.read(state).validate(order)
        if client:
            result = request(client, 'POST', base+'/orders', order)
            state = result.get('state') or request(client, 'GET', base+'/state')
        else:
            state = market.play(order)
        own = state['last_day']['you']
        profits.append(own['profit'])
        waste += own['meals']-own['sold']
        zero += own['sold'] == 0 and own['meals'] > 0
    if state['phase'] != 'over': raise RuntimeError('Practice did not complete')
    return {'policy': name, 'seed': seed, 'rivals': rivals, 'style': style, 'profit': state['you']['cash'],
            'waste': waste, 'zero_sale_days': zero, 'worst_day': min(profits),
            'decision_max_seconds': max(latencies), 'decision_mean_seconds': statistics.mean(latencies)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', action='store_true', help='Private practice only; never sends live orders')
    parser.add_argument('--seeds', nargs='+', default=['unseen-a', 'unseen-b'])
    parser.add_argument('--rivals', type=int, nargs='+', default=[3, 7])
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--styles', nargs='+', default=['mixed', 'crowd', 'moving'])
    parser.add_argument('--policies', nargs='+', default=['cautious', 'untrained', 'learned'])
    parser.add_argument('--output', default='data/evaluation-v1.json')
    args = parser.parse_args()
    client = None
    if args.server:
        load_env()
        client = Client()
    results = []
    for seed in args.seeds:
        for rivals in args.rivals:
            for style in (['house'] if client else args.styles):
                for name in args.policies:
                    result = run(name, seed, rivals, args.days, style, client)
                    results.append(result)
                    print(json.dumps(result), flush=True)
                    atomic_json(args.output, {'source': 'server-practice' if client else 'approximate-offline',
                                             'days': args.days, 'results': results})
    for name in args.policies:
        group = [r for r in results if r['policy'] == name]
        print('%s: average $%.2f, worst market $%.2f, maximum decision %.3fs' %
              (name, statistics.mean(r['profit'] for r in group), min(r['profit'] for r in group),
               max(r['decision_max_seconds'] for r in group)))


if __name__ == '__main__':
    main()
