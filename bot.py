"""Night Shift runner. Standard library only; importing this module never loads secrets."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import urllib.error
import urllib.request
from game import Rules, SPOTS, without_null_counts
from learning import Learner
from strategy import Brain
from history import warm_start
from forecast_log import append_unique
from mechanics import features
from politics import choose_vote, vote_analysis
from final_mode import attach_policy
from complaints import enabled as complaints_enabled, ENDPOINT as COMPLAINT_PATH
from coupon_play import enabled as coupon_enabled, decide as coupon_decide

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
    brain = Brain(learner=learner, samples=samples, small_crowds=small_crowds,
                  risk=.08 if small_crowds else .15,
                  spot_price_lift={'thompson': 1.39})
    attach_policy(brain)  # learns from every day; steers only in the final (see Runner.apply_profile)
    return brain


def fallback(state):
    """Bound loss if policy fails: assess all spots using forecast less rival stock."""
    state = without_null_counts(state)
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


STAFF_TRIES = 4   # a failed crew request is retried on later polls, a few times per night


class Runner:
    def __init__(self, client, brain, checkpoint=None, log=None, resume=True):
        self.client, self.brain = client, brain
        self.checkpoint, self.log = checkpoint, log
        self.accepted = None
        self.accepted_features = None
        self.accepted_order = None
        self.blocked = None
        self.pending_key = self.pending_order = None
        self.pending_features = None
        self.voted = set()
        self.staffed = set()
        self.staff_tries = {}
        self.complained = set()
        self.couponed = set()
        self.reported = set()
        self.force_final = False
        self.live_checkpoint = checkpoint
        checkpoint = self.resume_path(checkpoint)
        self.checkpoint = checkpoint
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
        post = diag.get('posture') or {}
        if post and (post.get('pressure') or post.get('lead')):
            print('  posture: pressure %.2f lead %.2f -> risk weight x%.2f, upside %.2f, roam bonus up to $%.0f' %
                  (post['pressure'], post['lead'], post['risk_scale'], post['upside'], post['explore']), flush=True)
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

    @staticmethod
    def resume_path(checkpoint):
        """After a restart during the final, resume the final checkpoint (newer than the live one)."""
        if not checkpoint:
            return checkpoint
        live = Path(checkpoint)
        final = live.parent / 'final-runtime-v1.json'
        try:
            if final.exists() and (not live.exists() or final.stat().st_mtime >= live.stat().st_mtime):
                return final
        except OSError:
            pass
        return live

    def profile_wanted(self, state):
        """'final' in final markets (or when forced for practice); 'baseline' otherwise or when killed."""
        if os.environ.get('NIGHTSHIFT_PROFILE') == 'baseline':
            return 'baseline'
        try:
            if json.loads((ROOT / 'data' / 'profile.json').read_text()).get('profile') == 'baseline':
                return 'baseline'
        except (OSError, ValueError, AttributeError):
            pass
        return 'final' if (state.get('mode') in ('final', 'showcase') or self.force_final) else 'baseline'

    def apply_profile(self, state):
        policy = getattr(self.brain, 'final_policy', None)
        if policy is None or not hasattr(self.brain, 'use_final'):
            return
        want = self.profile_wanted(state) == 'final'
        if want != getattr(self.brain, 'final_active', False):
            self.brain.use_final(want)
            print('Profile: %s' % ('FINAL (adaptive)' if want else 'baseline'), flush=True)
        if state.get('mode') in ('final', 'showcase') and hasattr(policy, 'reset_for_final') and policy.reset_for_final():
            print('Final start: tuner, competition and rule residuals reset; customer, rival, calibration and crew learning kept', flush=True)
        if self.live_checkpoint:  # final markets never write the live checkpoint, whatever the profile
            live = Path(self.live_checkpoint)
            self.checkpoint = live.parent / 'final-runtime-v1.json' if state.get('mode') in ('final', 'showcase') else live

    def final_extras(self, state, order):
        policy = getattr(self.brain, 'final_policy', None)
        if policy is None or not getattr(self.brain, 'final_active', False):
            return {}
        try:
            return policy.order_extras(state, order)
        except Exception:
            return {}

    def observe_safely(self, state):
        """Learning must never end the run: on any error skip that day's learning and keep polling."""
        try:
            self.brain.observe(state)
        except Exception as error:
            print('Day observation skipped (%s)' % type(error).__name__, flush=True)
            self.brain.seen = (state.get('last_day') or {}).get('day', getattr(self.brain, 'seen', None))

    def vote_report(self, state):
        """Say how the campaign looks for our order. Read-only; it never casts or changes a vote."""
        election = (state.get('politics') or {}).get('election') or {}
        key = (state.get('mode'), state.get('market'), election.get('id'), state.get('order_for_day'))
        order = self.accepted_order or getattr(self.brain, 'last_order', None)
        if not election.get('id') or key in self.reported or not order:
            return
        self.reported.add(key)
        try:
            analysis = vote_analysis(state, order, getattr(self.brain, 'game_day_rate', lambda: .25)(),
                                     getattr(self.brain, 'protest_rate', lambda: .30)())
            if analysis:
                parts = ['%s %+.0f/day, win %.0f%% (abstain %.0f%%), net %+.0f' %
                         (r['id'], r['daily_value'], 100*r['p_win_vote'], 100*r['p_win_abstain'], r['net'])
                         for r in analysis['rows']]
                print('  vote (ours: %s; %d unvoted; margin $%.0f): %s' %
                      (analysis['own'] or 'none', analysis['unvoted'], analysis['margin'], '; '.join(parts)), flush=True)
        except Exception as error:
            print('Vote report skipped (%s)' % type(error).__name__, flush=True)

    def vote(self, state, base):
        """Cast at most one cautious vote per election, after the order is safely handled."""
        if state.get('you') is None:
            return  # showcase: not a playing truck
        election = (state.get('politics') or {}).get('election') or {}
        key = (state.get('mode'), state.get('market'), election.get('id'))
        order = self.accepted_order or getattr(self.brain, 'last_order', None)
        if not election.get('id') or key in self.voted or not order:
            return
        self.voted.add(key)
        try:
            choice = choose_vote(state, order, getattr(self.brain, 'game_day_rate', lambda: .25)(),
                                 getattr(self.brain, 'protest_rate', lambda: .30)())
            if choice and not (state.get('you') or {}).get('vote'):  # never override a vote already cast
                status, _ = self.client.call('POST', base + '/vote', {'candidate': choice}, timeout=5.)
                print('Vote for %s: HTTP %s' % (choice, status), flush=True)
        except Exception as error:
            print('Vote skipped (%s)' % type(error).__name__, flush=True)

    def coupon(self, state, base):
        """Rarely drop a coupon on the rival we are racing (final only; see coupon_play.py)."""
        policy = getattr(self.brain, 'final_policy', None)
        key = (state.get('mode'), state.get('market'), state.get('order_for_day'))
        if (state.get('you') is None or policy is None or not getattr(self.brain, 'final_active', False)
                or not self.accepted_order or key in self.couponed or not coupon_enabled(state.get('mode'))):
            return
        self.couponed.add(key)
        try:
            decision = coupon_decide(state, self.brain.history, policy.adapter.coupon_cost(state) if policy.adapter else 20.)
            if not decision['drop']:
                return
            status, response = self.client.call('POST', base + '/coupons', {'target': decision['target']}, timeout=3.)
            print('  coupon dropped (HTTP %s) on %s: target loses about $%.0f for our $%.0f' %
                  (status, decision['target'], decision['loss'], decision['cost']), flush=True)
        except Exception as error:
            print('Coupon skipped (%s)' % type(error).__name__, flush=True)

    def complain(self, state, base):
        """Rarely file a health complaint (final only unless forced), after the order is accepted."""
        if state.get('you') is None:
            return  # showcase: not a playing truck
        policy = getattr(self.brain, 'final_policy', None)
        key = (state.get('mode'), state.get('market'), state.get('order_for_day'))
        if (policy is None or not getattr(self.brain, 'final_active', False) or not self.accepted_order
                or key in self.complained or not complaints_enabled(state.get('mode'))):
            return
        self.complained.add(key)
        try:
            decision = policy.complaint_decision(state, self.brain, self.accepted_order)
            if not decision.get('file'):
                return
            status, response = self.client.call('POST', base + COMPLAINT_PATH, {'target': decision['target']}, timeout=3.)
            caught = policy.complaints.record(state, decision['target'], response if status == 200 else {'error': status})
            print('  complaint filed (HTTP %s) against %s: %s%s' % (status, decision['target'], decision['reason'],
                  ' -- CAUGHT, banned' if caught else ''), flush=True)
        except Exception as error:
            print('Complaint skipped (%s)' % type(error).__name__, flush=True)

    def staff(self, state, base):
        """Set tomorrow's crew once per night, after the order is accepted."""
        if state.get('you') is None:
            return  # showcase: not a playing truck
        key = (state.get('mode'), state.get('market'), state.get('order_for_day'))
        order = self.accepted_order
        if not order or key in self.staffed or not hasattr(self.brain, 'crew_plan'):
            return
        tries = self.staff_tries.get(key, 0) + 1
        self.staff_tries[key] = tries
        if tries >= STAFF_TRIES:
            self.staffed.add(key)    # bounded: stop retrying this night
        try:
            plan = self.brain.crew_plan(state, order)
            if not plan:
                self.staffed.add(key)
                return
            note = 'unchanged'
            if plan['change']:
                status, _ = self.client.call('POST', base + '/staff',
                                             {'staff': plan['headcount'], 'wage': plan['wage']}, timeout=5.)
                note = 'set (HTTP %s)' % status
                if status != 200:
                    print('  crew not set (HTTP %s), attempt %d of %d' % (status, tries, STAFF_TRIES), flush=True)
                    return
            self.staffed.add(key)
            self.brain.diagnostics['crew'] = plan
            print('  crew %s: %d at $%.2f (was %d at $%.2f); need %d for %d meals; P(enough) %.0f%%; '
                  'expected wages $%.0f; poach %.1f%%/exposed rival, %d exposed; pool ask median %s, %s unemployed' %
                  (note, plan['headcount'], plan['wage'], plan['current'][0], plan['current'][1],
                   plan['needed'], order['meals'], 100*plan['prob_enough_staff'], plan['expected_daily_wages'],
                   100*plan['poach_rate_per_exposed_rival'], plan['rivals_exposed'],
                   '%.2f' % plan['pool_median_ask'] if plan['pool_median_ask'] else '?',
                   plan['unemployed']), flush=True)
            checks = self.brain.crew.checks[-1:]
            if checks:
                c = checks[0]
                print('  crew check day %d: predicted %.1f employees, actual %d, poached %d' %
                      (c['day'], c['predicted_employees'], c['actual_employees'], c['poached']), flush=True)
        except Exception as error:
            print('Crew plan skipped (%s)' % type(error).__name__, flush=True)

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
                self.apply_profile(state)
                if state.get('phase') == 'over' and state.get('mode') == 'practice':
                    self.brain.observe(state)
                    self.record_evaluation(state)
                    self.save()
                    return state
                key = [state.get('mode'), state.get('market'), state.get('order_for_day')]
                active_features = features(state.get('features'))
                same_features = (active_features == self.accepted_features or
                                 self.accepted_features is None and not any(active_features.values()))
                if (state.get('phase') != 'night' or not state.get('orders_open') or state.get('you') is None
                        or key == self.blocked or key == self.accepted and same_features):
                    last = state.get('last_day') or {}
                    if state.get('mode') in ('live', 'final', 'showcase') and last.get('day') is not None and (
                            getattr(self.brain, 'seen', None) != last['day'] or
                            getattr(self.brain, 'market', None) != key[:2]):
                        self.observe_safely(state)
                        self.archive(state)
                        self.save()
                    self.record_evaluation(state)
                    self.health(state)
                    self.vote_report(state)
                    self.vote(state, base)
                    self.staff(state, base)
                    self.complain(state, base)
                    self.coupon(state, base)
                    continue
                if key != self.pending_key or active_features != self.pending_features:
                    try:
                        order = self.brain.decide(state)
                    except Exception as error:
                        where = traceback.extract_tb(error.__traceback__)[-1]
                        print('Policy error (%s: %s at %s:%d); using bounded fallback' %
                              (type(error).__name__, str(error)[:70], Path(where.filename).name, where.lineno), flush=True)
                        order = fallback(state)
                    self.pending_key, self.pending_order = key, order
                    self.pending_features = active_features
                order = dict(self.pending_order, **self.final_extras(state, self.pending_order))
                policy = getattr(self.brain, 'final_policy', None)
                if getattr(self.brain, 'final_active', False) and policy.is_wrecked(state):
                    self.blocked = key  # wrecked in a riot: orders would get 403
                    continue
                # Submission precedes logging/checkpoint disk I/O in the two-second window.
                status, response = self.client.call('POST', base + '/orders', order, timeout=1.)
                if status == 200:
                    self.accepted = key
                    self.accepted_features = active_features
                    self.accepted_order = order
                    self.record(state, order)
                    self.save()
                    self.health(state)
                    self.vote_report(state)
                    self.vote(state, base)
                    self.staff(state, base)
                    self.complain(state, base)
                    self.coupon(state, base)
                    if isinstance(response.get('state'), dict):
                        next_state = response['state']
                elif status == 409:
                    # Read immediately: a market may have advanced while we computed.
                    version = 0
                    self.pending_key = None
                    time.sleep(.05)
                elif status in (401, 404):
                    raise RuntimeError('Order request denied (HTTP %s)' % status)
                elif status == 403:
                    # Not allowed tonight (for example wrecked in a riot): ask once, then wait for new state.
                    self.blocked = key
                    print('Order refused (HTTP 403): %s' % response.get('error', ''), flush=True)
                    time.sleep(1.)
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
    parser.add_argument('--profile', choices=('auto', 'final', 'baseline'), default='auto',
                        help='final: use the adaptive policy even in practice; auto: final markets only')
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
    if args.profile == 'baseline':
        os.environ['NIGHTSHIFT_PROFILE'] = 'baseline'
    runner = Runner(client, brain, checkpoint,
                    ROOT / 'data' / ('practice-v1.jsonl' if args.practice else 'decisions-v1.jsonl'),
                    resume=not args.fresh)
    runner.force_final = args.profile == 'final'
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
