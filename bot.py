"""Night Shift runner. Standard library only; importing this module never loads secrets."""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from game import Rules, SPOTS
from learning import Learner
from strategy import Brain
from history import warm_start
from forecast_log import append_unique
from mechanics import features

ROOT = Path(__file__).resolve().parent


def load_env():
    path = ROOT / '.env'
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith('export '):
                line = line[7:]
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip().strip('\"\''))


class Client:
    def __init__(self, url=None, key=None):
        self.url = (url or os.environ.get('NIGHTSHIFT_URL', 'https://nightshift.joinbuilders.org')).rstrip('/')
        self.key = key or os.environ.get('NIGHTSHIFT_KEY', '')
        if not self.key:
            raise ValueError('Set NIGHTSHIFT_KEY in .env or the environment')
        if not self.url.startswith('https://'):
            raise ValueError('NIGHTSHIFT_URL must use HTTPS')

    def call(self, method, path, body=None, timeout=30):
        if not path.startswith('/api/v1/') and path != '/api/v1':
            raise ValueError('Unexpected API path')
        request = urllib.request.Request(self.url + path, method=method,
                    data=None if body is None else json.dumps(body).encode(),
                    headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            try:
                return error.code, json.load(error)
            except ValueError:
                return error.code, {'error': 'HTTP request failed'}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')
    temporary.replace(path)


def make_brain(model=None, samples=32, small_crowds=True):
    path = Path(model) if model else ROOT / 'model' / 'prior.json'
    learner = Learner.load(path) if path.exists() else Learner()
    # Inventory still pays full ingredient/waste costs in expected profit; ease
    # only the additional downside penalty to permit supported demand growth.
    return Brain(learner=learner, samples=samples, small_crowds=small_crowds,
                 risk=.08 if small_crowds else .15)


def fallback(state):
    """Bound loss if policy fails: assess all spots using forecast less rival stock."""
    rules = Rules.read(state)
    forecast = (state.get('forecast') or {}).get('expected_customers') or {}
    last = (state.get('last_day') or {}).get('spots') or {}
    def opportunity(spot):
        rivals = (last.get(spot) or {}).get('trucks') or []
        stock = sum(t.get('meals', 0) for t in rivals if t.get('id') != state.get('you', {}).get('id'))
        return max(0., float(forecast.get(spot, 0)) - stock)
    bottom_two = sorted(rules.spots, key=lambda spot: (float(forecast.get(spot, 0)), spot))[:2]
    spot = max(bottom_two, key=opportunity)
    meals = min(30, rules.capacity('standard'), int(opportunity(spot) * .35))
    return rules.validate({'spot': spot, 'price': max(rules.price_min, min(rules.price_max, 7.0)),
                           'quality': 'standard', 'meals': max(0, meals)})


class Runner:
    def __init__(self, client, brain, checkpoint=None, log=None, resume=True):
        self.client, self.brain = client, brain
        self.checkpoint, self.log = checkpoint, log
        self.accepted = None
        self.accepted_features = None
        self.pending_key = self.pending_order = None
        self.pending_features = None
        if resume and checkpoint and Path(checkpoint).exists():
            try:
                saved = json.loads(Path(checkpoint).read_text())
                self.brain.restore(saved['brain'])
                self.accepted = saved['accepted']
                self.accepted_features = saved.get('accepted_features')
            except (ValueError, KeyError, TypeError):
                print('Checkpoint unreadable; starting from the trained model', flush=True)

    def save(self):
        if self.checkpoint:
            try:
                atomic_json(self.checkpoint, {'accepted': self.accepted,
                                             'accepted_features': self.accepted_features,
                                             'brain': self.brain.dump()})
            except OSError:
                print('Checkpoint write failed', flush=True)

    def archive(self, state):
        if self.log and state.get('mode') in ('live', 'final') and (state.get('last_day') or {}).get('day') is not None:
            try:
                name = '%s-%d-day%02d.json' % (state['mode'], int(state.get('market') or 1), state['last_day']['day'])
                atomic_json(Path(self.log).parent / 'live-observations' / name, state)
            except (OSError, ValueError, TypeError):
                print('Live observation archive failed', flush=True)

    def health(self, state):
        if self.checkpoint:
            try:
                atomic_json(Path(self.checkpoint).parent / 'connection-status.json',
                            {'received_at': time.time(), 'mode': state.get('mode'),
                             'market': state.get('market'), 'day': state.get('day'),
                             'phase': state.get('phase'), 'order_for_day': state.get('order_for_day'),
                             'features': features(state.get('features')),
                             'accepted': self.accepted, 'learned_day': getattr(self.brain, 'seen', None)})
            except OSError:
                pass

    def record(self, state, order):
        self.archive(state)
        self.record_evaluation(state)
        if self.log:
            try:
                Path(self.log).parent.mkdir(parents=True, exist_ok=True)
                with open(self.log, 'a') as out:
                    out.write(json.dumps({'state': state, 'order': order, 'decision': self.brain.diagnostics}) + '\n')
                frozen = getattr(self.brain, 'forecast_record', None)
                if frozen and frozen['order'] == order:
                    prefix = 'practice-' if state.get('mode') == 'practice' else ''
                    append_unique(Path(self.log).parent/(prefix+'forecasts-v2.jsonl'),
                                  dict(frozen, accepted_at=time.time(), accepted=True))
            except OSError:
                print('Decision log write failed', flush=True)
        diag = self.brain.diagnostics
        if diag.get('forecast_capture_error'):
            print('Forecast evidence unavailable (%s)' % diag['forecast_capture_error'], flush=True)
        for result in diag.get('ranking', []):
            candidate = result['order']
            print('  %-9s %-8s $%5.2f x%3d predicted sales %.1f profit $%.0f' %
                  (candidate['spot'], candidate['quality'], candidate['price'], candidate['meals'],
                   result['sales'], result['profit']), flush=True)
        print('  chose %s; search %.3fs' % (order, diag.get('elapsed', 0)), flush=True)

    def record_evaluation(self, state):
        if not self.log:
            return
        last = state.get('last_day') or {}
        for check in getattr(self.brain, 'prediction_checks', []):
            if (check.get('forecast_id') and check.get('day') == last.get('day')
                    and check.get('mode') == state.get('mode') and check.get('market') == state.get('market')):
                try:
                    prefix = 'practice-' if state.get('mode') == 'practice' else ''
                    append_unique(Path(self.log).parent/(prefix+'evaluation-v2.jsonl'),
                                  dict(check, schema_version=2, observed_at=time.time(),
                                       actual_result=last.get('you'), actual_spots=last.get('spots')))
                except OSError:
                    print('Evaluation log write failed', flush=True)

    def play(self, base='/api/v1'):
        version, next_state, backoff = 0, None, .25
        while True:
            try:
                if next_state is None:
                    status, state = self.client.call('GET', base + '/state?since=%s' % version)
                    if status in (401, 404):
                        raise RuntimeError('State request denied (HTTP %s)' % status)
                    if status != 200:
                        if status == 429:
                            time.sleep(1.)
                        else:
                            time.sleep(backoff)
                            backoff = min(5., backoff * 2)
                        continue
                else:
                    state, next_state = next_state, None
                version = state['version']
                backoff = .25
                if state.get('phase') == 'over' and state.get('mode') == 'practice':
                    self.brain.observe(state)
                    self.record_evaluation(state)
                    self.save()
                    return state
                key = [state.get('mode'), state.get('market'), state.get('order_for_day')]
                active_features = features(state.get('features'))
                same_features = (active_features == self.accepted_features or
                                 self.accepted_features is None and not any(active_features.values()))
                if (state.get('phase') != 'night' or not state.get('orders_open')
                        or key == self.accepted and same_features):
                    last = state.get('last_day') or {}
                    if state.get('mode') in ('live', 'final') and last.get('day') is not None and (
                            getattr(self.brain, 'seen', None) != last['day'] or
                            getattr(self.brain, 'market', None) != key[:2]):
                        self.brain.observe(state)
                        self.archive(state)
                        self.save()
                    self.record_evaluation(state)
                    self.health(state)
                    continue
                if key != self.pending_key or active_features != self.pending_features:
                    try:
                        order = self.brain.decide(state)
                    except Exception as error:
                        print('Policy error (%s); using bounded fallback' % type(error).__name__, flush=True)
                        order = fallback(state)
                    self.pending_key, self.pending_order = key, order
                    self.pending_features = active_features
                order = self.pending_order
                # Submission precedes logging/checkpoint disk I/O in the two-second window.
                status, response = self.client.call('POST', base + '/orders', order, timeout=1.)
                if status == 200:
                    self.accepted = key
                    self.accepted_features = active_features
                    self.record(state, order)
                    self.save()
                    self.health(state)
                    if isinstance(response.get('state'), dict):
                        next_state = response['state']
                elif status == 409:
                    # Read immediately: a market may have advanced while we computed.
                    version = 0
                    self.pending_key = None
                    time.sleep(.05)
                elif status in (401, 404):
                    raise RuntimeError('Order request denied (HTTP %s)' % status)
                elif status == 400:
                    # Avoid a tight loop; one validated conservative replacement.
                    self.pending_order = fallback(state)
                    version = 0
                    time.sleep(.2)
                else:
                    version = 0
                    time.sleep(1. if status == 429 else .1)
            except (urllib.error.URLError, OSError, ValueError, KeyError) as error:
                # Poll from zero after an uncertain submission; retry the same order,
                # without learning the same day twice or waiting 25 seconds for a change.
                version = 0
                print('Connection/state error (%s); retrying' % type(error).__name__, flush=True)
                time.sleep(backoff)
                backoff = min(5., backoff * 2)


def main():
    parser = argparse.ArgumentParser(description='Learned Night Shift profit optimizer')
    parser.add_argument('--practice', action='store_true')
    parser.add_argument('--predict', action='store_true', help='Read live state and save a prediction without submitting')
    parser.add_argument('--seed', default='rebuild-check')
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--rivals', type=int, default=3)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--fresh', action='store_true', help='Ignore live checkpoint')
    args = parser.parse_args()
    if args.predict and args.practice:
        parser.error('--predict is for a read-only live forecast; omit --practice')
    load_env()
    client = Client()
    brain = make_brain(args.model, args.samples)
    base = '/api/v1'
    checkpoint = None if args.practice else ROOT / 'data' / 'runtime-v1.json'
    if args.practice:
        status, response = client.call('POST', base + '/practice',
                            {'seed': args.seed, 'days': args.days, 'rivals': args.rivals})
        if status not in (200, 201):
            raise RuntimeError('Practice creation failed (HTTP %s)' % status)
        base = response['base']
    runner = Runner(client, brain, checkpoint,
                    ROOT / 'data' / ('practice-v1.jsonl' if args.practice else 'decisions-v1.jsonl'),
                    resume=not args.fresh)
    if not args.practice:
        loaded = warm_start(runner.brain, ROOT)
        print('Loaded %d additional saved live days' % loaded, flush=True)
    if args.predict:
        status, state = client.call('GET', '/api/v1/state?since=0')
        if status != 200:
            raise RuntimeError('State request failed (HTTP %s)' % status)
        if state.get('order_for_day') is None:
            state = dict(state, order_for_day=(state.get('forecast') or {}).get('day') or state.get('day', 0)+1)
        order = runner.brain.decide(state)
        atomic_json(ROOT / 'data' / 'next-prediction.json',
                    {'mode': state.get('mode'), 'market': state.get('market'),
                     'order_for_day': state.get('order_for_day'), 'submitted': False,
                     'order': order, 'diagnostics': runner.brain.diagnostics})
        print('Prediction only; no order submitted:', order)
        return
    final = runner.play(base)
    if final:
        print('Practice complete: $%.2f' % final['you']['cash'])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped')
