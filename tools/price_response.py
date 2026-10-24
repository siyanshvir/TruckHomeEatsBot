"""Read-only: predicted versus actual sales by price, from the local evaluation ledger.

Shows whether the model over-predicts as prices rise. It changes no policy and
makes no API request. Sellouts are lower bounds on demand, so they are listed apart.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(path=None):
    path = Path(path) if path else ROOT / 'data' / 'evaluation-v2.jsonl'
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get('mode') == 'live' and row.get('order_matches') and not row.get('surprise'):
            rows.append(row)
    rows.sort(key=lambda r: r['day'])
    print('day spot      quality  price meals  predicted actual  error  note')
    for r in rows:
        order = r['order']
        error = r['actual_sales'] - r['expected_sales']
        print('%3d %-9s %-8s %5.2f %5d  %9.1f %6.0f %+6.1f  %s' %
              (r['day'], order['spot'], order['quality'], order['price'], order['meals'],
               r['expected_sales'], r['actual_sales'], error, 'sold out (demand >= actual)' if r['sellout'] else ''))
    open_rows = [r for r in rows if not r['sellout']]
    if open_rows:
        bias = sum(r['actual_sales'] - r['expected_sales'] for r in open_rows) / len(open_rows)
        print('\nMean error on days that did not sell out: %+.1f meals over %d days' % (bias, len(open_rows)))
    print('Days that sold out understate demand, so they cannot show over-prediction.')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
