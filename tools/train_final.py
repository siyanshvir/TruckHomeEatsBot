"""Fit final-bot calibration and controller priors from chronological local data.

    python3 tools/train_final.py --data-root data --output model/final_priors.json

Sellouts are lower bounds on demand, so they never enter the mean sales ratio; they are counted.
Never overwrites model/prior.json. Output is deterministic for identical input.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bot import atomic_json
from dataset import _read_lines, calibration_rows, iter_days

RATIO_BINS = (0.0, .5, .6, .7, .8, .9, 1.0, 1.25, 10.0)  # price / value ceiling
ARMS = ('0.9', '1.0', '1.1')
KNOBS = ('price_aggr', 'breadth', 'risk')
PROTECTED = 'prior.json'


def bin_label(ratio):
    for low, high in zip(RATIO_BINS, RATIO_BINS[1:]):
        if ratio < high:
            return '%.2f-%.2f' % (low, high)
    return '%.2f-%.2f' % (RATIO_BINS[-2], RATIO_BINS[-1])


def _stats(values, censored):
    mean = round(sum(values)/len(values), 6) if values else 1.0
    return {'mean': mean, 'n': len(values), 'censored': censored}


def calibration(rows):
    """Mean actual/expected sales ratio overall and by quality x price-ratio bin."""
    usable = [r for r in rows if r['expected_sales'] > 1e-9]
    full = [r for r in usable if not r['sellout']]
    groups, censored = {}, {}
    for r in usable:
        key = (r['quality'], bin_label(r['price_ratio']))
        if r['sellout']:
            censored[key] = censored.get(key, 0)+1
        else:
            groups.setdefault(key, []).append(r['actual_sales']/r['expected_sales'])
    buckets = {}
    for quality, label in sorted(set(groups) | set(censored)):
        buckets.setdefault(quality, {})[label] = _stats(groups.get((quality, label), []), censored.get((quality, label), 0))
    ratios = [r['actual_sales']/r['expected_sales'] for r in full]
    return {'global': dict(_stats(ratios, len(usable)-len(full)),
                           ratio_bins=['%.2f-%.2f' % (a, b) for a, b in zip(RATIO_BINS, RATIO_BINS[1:])]),
            'buckets': buckets}


def controller():
    """Neutral Gaussian arm priors: data to move them does not exist yet."""
    return {knob: {arm: {'mean': 0, 'var': 1, 'n': 0} for arm in ARMS} for knob in KNOBS}


def build(data_root):
    days = list(iter_days(data_root))
    sources = {}
    for day in days:
        sources[day['mode']] = sources.get(day['mode'], 0)+1
    rows = calibration_rows(day for day in days if 'prediction' in day)
    # Evaluation rows without a matching observation (eg. observation file missing) still count.
    matched = {(d['mode'], d['market'], d['day']) for d in days if 'prediction' in d}
    extra = []
    path = Path(data_root) / 'evaluation-v2.jsonl'
    if path.exists():
        extra = [r for r in _read_lines(path) if (r.get('mode'), r.get('market'), r.get('day')) not in matched]
    rows += calibration_rows(extra)
    return {'version': 1, 'calibration': calibration(rows), 'controller': controller(),
            'meta': {'days': len(days), 'source_modes': dict(sorted(sources.items())),
                     'calibration_rows': len(rows)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--output', default='model/final_priors.json')
    args = parser.parse_args(argv)
    if Path(args.output).name == PROTECTED:
        parser.error('refusing to write %s; choose another output' % PROTECTED)
    priors = build(args.data_root)
    atomic_json(args.output, json.loads(json.dumps(priors, sort_keys=True)))
    g = priors['calibration']['global']
    print('days=%d modes=%s calibration rows=%d (full %d, sellouts %d) mean sales ratio=%.3f -> %s' %
          (priors['meta']['days'], priors['meta']['source_modes'], priors['meta']['calibration_rows'],
           g['n'], g['censored'], g['mean'], args.output))
    return priors


if __name__ == '__main__':
    main()
