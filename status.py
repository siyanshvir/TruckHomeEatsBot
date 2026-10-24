"""Read local bot health and prediction checks; no credentials or API requests."""
import json
from pathlib import Path
import time


def main():
    root = Path(__file__).resolve().parent / 'data'
    def read(name):
        try:
            return json.loads((root / name).read_text())
        except (OSError, ValueError):
            return {}
    health = read('connection-status.json')
    if health:
        age = max(0., time.time()-health['received_at'])
        print('Last successful API state: %.0fs ago%s' %
              (age, ' — stale; check the bot terminal' if age > 60 else ''))
        print('Market %s / %s; day %s; phase %s' %
              (health['mode'], health['market'], health['day'], health['phase']))
        if 'features' in health:
            print('Active rule flags:', health['features'])
    else:
        print('No connection status yet; restart the updated bot to begin recording it.')
    saved = read('runtime-v1.json')
    brain = saved.get('brain', {})
    print('Learned through day:', brain.get('seen'))
    print('Last accepted [mode, market, day]:', saved.get('accepted'))
    print('Last planned order:', brain.get('last_order'))
    checks = brain.get('prediction_checks', [])[-5:]
    if not checks:
        print('Prediction checks appear after the first day planned by this version finishes.')
    for c in checks:
        print('Day %s: sales predicted %.1f / actual %.1f; profit predicted $%.2f / actual %s; '
              'rivals predicted %.1f / actual %s%s' %
              (c['day'], c['expected_sales'], c['actual_sales'], c['expected_profit'],
               c['actual_profit'], c['expected_rivals'], c['actual_rivals'],
               ' (order differed; excluded from demand error)' if not c['order_matches'] else
               ' (sold out; demand is a lower bound)' if c['sellout'] else ''))


if __name__ == '__main__':
    main()
