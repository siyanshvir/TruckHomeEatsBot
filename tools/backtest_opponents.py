"""Walk-forward rival prediction scoring; today's outcomes are never input features."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bot import atomic_json
from game import identity
from learning import training_rows
from strategy import Brain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/collect.jsonl')
    parser.add_argument('--output', default='data/opponent-validation.json')
    args = parser.parse_args()
    held = set(json.loads(Path('model/training-report.json').read_text())['held_out_markets'])
    grouped = {}
    for row in training_rows(args.data):
        if str(row['seed']) in held:
            grouped.setdefault(str(row['seed']), []).append(row)
    total = {'scored': 0, 'log_loss': 0., 'repeat_log_loss': 0., 'price_count': 0,
             'price_error': 0., 'repeat_price_error': 0.}
    for seed, rows in sorted(grouped.items()):
        brain = Brain()
        for row in sorted(rows, key=lambda r: r['last_day']['day']):
            pre, last = row['pre'], row['last_day']
            # The pre record is the historical forecast available BEFORE this outcome.
            brain.pending = {'day': last['day'], 'forecast': pre['forecast']}
            # Set market first so observe won't clear this known historical forecast.
            brain.market = ['practice', seed]
            roster = [{'id': t['id']} for t in pre['trucks']]
            brain.observe({'mode': 'practice', 'market': seed, 'you': {'id': row['me']},
                           'last_day': last, 'trucks': roster})
        for value in brain.opponents.data.values():
            for key in total: total[key] += value[key]
    report = {'method': 'predict-then-observe, fresh rival model per held-out market',
              'markets': sorted(grouped), 'location_predictions': total['scored'],
              'location_log_loss': total['log_loss']/max(1, total['scored']),
              'repeat_location_log_loss': total['repeat_log_loss']/max(1, total['scored']),
              'price_predictions': total['price_count'],
              'price_mae': total['price_error']/max(1, total['price_count']),
              'repeat_price_mae': total['repeat_price_error']/max(1, total['price_count']),
              'limitation': 'House bots are often stationary; this is not evidence of live opponent accuracy.'}
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
