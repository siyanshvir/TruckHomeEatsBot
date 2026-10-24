"""Train on saved practice pairs; split entire markets before validation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bot import atomic_json
from game import customer_model, events, expected_sales, identity, spot_results
from learning import Learner, PARTICLES, train, training_rows


def error(learner, rows):
    errors, censored = [], 0
    for row in rows:
        reps = {identity(t): t.get('rep') or {} for t in row['pre'].get('trucks', [])}
        for spot, info in spot_results(row['last_day']).items():
            records = info.get('trucks') or []
            if not records:
                continue
            trucks = [(t['price'], t['quality'], reps.get(identity(t), {}).get(spot, 50), t['meals']) for t in records]
            predictions = [0.] * len(trucks)
            for weight, particle in zip(learner.posterior(spot), PARTICLES):
                sold = expected_sales(trucks, info['customers'], customer_model(spot, events(row['last_day'].get('events')), particle))
                for i, value in enumerate(sold):
                    predictions[i] += weight * value
            for t, value in zip(records, predictions):
                if t['sold'] >= t['meals']:
                    errors.append(max(0., t['sold'] - value))
                    censored += 1
                else:
                    errors.append(abs(value - t['sold']))
    return {'mean_sales_error': sum(errors)/max(1, len(errors)), 'observations': len(errors), 'sellouts': censored}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/collect.jsonl')
    parser.add_argument('--output', default='model/prior.json')
    args = parser.parse_args()
    rows = list(training_rows(args.data))
    seeds = sorted({str(r['seed']) for r in rows})
    if len(seeds) < 2:
        raise ValueError('Need at least two independent markets for holdout validation')
    held = set(seeds[::5])
    fit = [r for r in rows if str(r['seed']) not in held]
    test = [r for r in rows if str(r['seed']) in held]
    learner = train(fit)
    report = {'training_days': len(fit), 'held_out_days': len(test), 'held_out_markets': sorted(held),
              'untrained': error(Learner(), test), 'trained': error(learner, test)}
    # Ship a model trained only on the fitting partition; holdout stays unseen.
    atomic_json(args.output, learner.to_dict())
    atomic_json(str(Path(args.output).with_name('training-report.json')), report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
