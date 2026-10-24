"""Offline rival-offer candidate. Never submit orders or update live checkpoints.

Preserves production location/price/inventory criteria. Candidate changes only:
condition rival offers on their sampled location when two historical observations
exist there, and use exact public reputation. This is not validated for promotion.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot import make_brain
from game import QUALITIES, clamp, customer_model, events, identity, weather_factor, weather_probs
from strategy import Brain


class ShadowIncumbentBrain(Brain):
    def offer_history(self, history, spot):
        local = [record for record in history if record['spot'] == spot]
        return local if len(local) >= 2 else history

    def scenarios(self, state, rules, counts, rng):
        me = identity(state.get('you') or {})
        roster = [identity(t) for t in state.get('trucks') or [] if identity(t) != me]
        public = {identity(t): t.get('reputation') or {} for t in state.get('trucks') or []}
        fc = state.get('forecast') or {}
        odds, kinds = weather_probs(fc), events(fc.get('events'))
        field = [h[-1] for name, h in self.history.items() if name in roster and h]
        scenarios = []
        self.scenario_rivals = []
        for index in range(self.samples):
            target, acc = (index + .5) / self.samples, 0.
            weather = list(odds)[-1]
            for w, probability in sorted(odds.items()):
                acc += probability
                if target <= acc:
                    weather = w
                    break
            placements = {s: [] for s in rules.spots}
            active = 0.
            named_offers = []
            for name in roster:
                history = self.history.get(name, [])
                if rng.random() > self.opponents.attendance(name):
                    continue
                weights = self.opponents.placement(name, history, fc, self.history, rules.spots)
                spot = rng.choices(rules.spots, weights=weights, k=1)[0]
                conditional = self.offer_history(history, spot)
                if conditional:
                    q = rng.choices(QUALITIES, weights=self.opponents.quality(conditional, fc), k=1)[0]
                    mean, spread, stock = self.opponents.pricing(name, conditional, fc, self.history, q, spot)
                    price = rng.gauss(mean, spread)
                    meals = max(0, min(rules.capacity(q), int(stock * rng.uniform(.85, 1.15))))
                else:
                    record = rng.choice(field) if field else None
                    q = record['quality'] if record else rng.choice(QUALITIES)
                    price = record['price'] if record else {'cheap': 6.5, 'standard': 9., 'premium': 10.}[q]
                    price += rng.gauss(0, .7)
                    meals = min(rules.capacity(q), int(record['meals']) if record else 80)
                price = round(clamp(price, rules.price_min, rules.price_max), 2)
                rep = public.get(name, {}).get(spot, self.reputations.get(name, {}).get(spot, 50.))
                placements[spot].append((price, q, clamp(rep, 0, 100), meals))
                named_offers.append({'id': name, 'spot': spot, 'price': price, 'quality': q,
                                     'reputation': clamp(rep, 0, 100), 'meals': meals})
                active += min(1., meals * rules.costs[q] / 250.)
            crowds, models = {}, {}
            for spot in rules.spots:
                average = sum(p * weather_factor(w, spot) for w, p in odds.items())
                conditional_weather = weather_factor(weather, spot) / max(average, .1)
                width = math.sqrt(3.) * self.learner.crowd_noise(spot, weather)
                crowds[spot] = (counts[spot] * conditional_weather * self.learner.crowd_multiplier(spot, weather)
                                * rng.uniform(1-width, 1+width))
                models[spot] = customer_model(spot, kinds, self.learner.sample(spot, rng))
            scenarios.append((placements, crowds, models, active))
            self.scenario_rivals.append(named_offers)
        return scenarios


def replay_before(brain, states, target_day):
    """No result on or after target_day can enter the training history."""
    for state in sorted(states, key=lambda s: (s.get('last_day') or {}).get('day', 0)):
        day = (state.get('last_day') or {}).get('day')
        if day is None or day >= target_day:
            continue
        brain.observe(state)
        forecast = state.get('forecast') or {}
        next_day = forecast.get('day') or state.get('order_for_day')
        if next_day is not None and next_day > day:
            brain.pending = {'day': next_day, 'forecast': forecast}
            if 'features' in state:
                brain.pending['features'] = state['features']


def frozen_history(root, target_day):
    states = []
    for path in (root/'data'/'live-observations').glob('live-*-day*.json'):
        match = re.search(r'-day(\d+)\.json$', path.name)
        if match and int(match.group(1)) < target_day:
            states.append(json.loads(path.read_text()))
    return states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--day', type=int, default=14)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--samples', type=int, default=32)
    args = parser.parse_args()
    decisions = [json.loads(line) for line in (ROOT/'data'/'decisions-v1.jsonl').read_text().splitlines()]
    matches = [r for r in decisions if r['state'].get('order_for_day') == args.day]
    if not matches:
        parser.error('No frozen pre-decision state for requested day')
    state = matches[-1]['state']
    if state.get('mode') != 'live':
        parser.error('Expected a saved live pre-decision state')
    history = [s for s in frozen_history(ROOT, args.day)
               if s.get('mode') == state.get('mode') and s.get('market') == state.get('market')]
    rows = []
    for seed in args.seeds:
        for policy, cls in [('production', Brain), ('candidate', ShadowIncumbentBrain)]:
            initial = make_brain(samples=args.samples)
            brain = cls(learner=initial.learner, seed=seed, samples=args.samples,
                        risk=initial.risk, small_crowds=True)
            replay_before(brain, copy.deepcopy(history), args.day)
            order = brain.decide(copy.deepcopy(state))
            chosen = brain.diagnostics['chosen']
            rows.append({'policy': policy, 'seed': seed, 'order': order,
                         'predicted_sales': chosen['sales'], 'predicted_profit': chosen['profit'],
                         'predicted_unsold': chosen['expected_unsold'],
                         'risk': brain.diagnostics.get('risk')})
    summary = {policy: {'runs': len([r for r in rows if r['policy'] == policy]),
                       'mean_meals': sum(r['order']['meals'] for r in rows if r['policy'] == policy)/len(args.seeds),
                       'mean_predicted_profit': sum(r['predicted_profit'] for r in rows if r['policy'] == policy)/len(args.seeds)}
               for policy in ('production', 'candidate')}
    print(json.dumps({'day': args.day, 'training_days': sorted(s['last_day']['day'] for s in history),
                      'warning': 'Offline predictions only. Different model predictions are not realized profit or validation.',
                      'runs': rows, 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
