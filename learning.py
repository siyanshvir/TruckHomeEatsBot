"""Online Bayesian model averaging, shared by training and the playing policy."""
import copy
import json
import math
from pathlib import Path
from game import (SPOTS, QUALITIES, REP_CHANGE, PRIORS, Rules, clamp, customer_model,
                  events, expected_sales, identity, spot_results, weather_factor, weather_probs)

# Broad enough to represent uncertainty rather than fitting one guessed distribution.
PARTICLES = [(mu, sd, taste) for mu in (-3., -1.5, 0., 1.5, 3.)
             for sd in (.65, 1., 1.4) for taste in (-.12, 0., .12)]


FEATURE_NAMES = ('fair_choice', 'walk_over', 'salvage')


def feature_tuple(features):
    """Canonical realized engine flags; never read tomorrow's forecast."""
    if features is None:
        return (False, False, False)
    if isinstance(features, (tuple, list)) and len(features) == len(FEATURE_NAMES):
        return tuple(bool(flag) for flag in features)
    if not isinstance(features, dict):
        raise ValueError('Expected feature flags as a mapping or three-element sequence')
    def enabled(value):
        return bool(value.get('enabled', False)) if isinstance(value, dict) else bool(value)
    return tuple(enabled(features.get(name, False)) for name in FEATURE_NAMES)


class Learner:
    VERSION = 1

    def __init__(self):
        self.active_features = (False, False, False)
        self.regimes = {}
        self._bind(self._fresh_bank())
        self.regimes[self._regime_key(self.active_features)] = self._current_bank()

    @staticmethod
    def _fresh_bank():
        return {'weights': {s: [0.] * len(PARTICLES) for s in SPOTS},
                'counts': {s: 0 for s in SPOTS}, 'crowd': {s: {} for s in SPOTS},
                'crowd_variance': {s: {} for s in SPOTS}, 'rows': 0}

    @staticmethod
    def _regime_key(features):
        return ''.join('1' if flag else '0' for flag in features)

    def _current_bank(self):
        return {name: getattr(self, name) for name in
                ('weights', 'counts', 'crowd', 'crowd_variance', 'rows')}

    def _bind(self, bank):
        for name, value in bank.items():
            setattr(self, name, value)

    def activate(self, features):
        """Select a rules-specific posterior, retaining all other feature regimes.

        New engine behavior starts with the broad customer prior, rather than
        transferring likelihoods fitted under different customer-choice rules.
        Rival history lives outside Learner and is unaffected.
        """
        flags = feature_tuple(features)
        if flags == self.active_features:
            return False
        self.regimes[self._regime_key(self.active_features)] = self._current_bank()
        key = self._regime_key(flags)
        if key not in self.regimes:
            self.regimes[key] = self._fresh_bank()
        self._bind(self.regimes[key])
        self.active_features = flags
        return True

    def to_dict(self):
        # Detach serialized snapshots from both live state and other banks.
        banks = dict(self.regimes)
        banks[self._regime_key(self.active_features)] = self._current_bank()
        snapshot = copy.deepcopy(self._current_bank())
        snapshot.update(version=self.VERSION, active_features=list(self.active_features),
                        regimes=copy.deepcopy(banks))
        return snapshot

    @classmethod
    def from_dict(cls, data):
        if data.get('version') != cls.VERSION:
            raise ValueError('Unsupported learned-model version')
        def read_bank(raw):
            bank = cls._fresh_bank()
            bank.update({name: copy.deepcopy(raw[name]) for name in bank if name in raw})
            for spot in SPOTS:
                weights = bank['weights'][spot]
                if len(weights) != len(PARTICLES) or not all(math.isfinite(x) for x in weights):
                    raise ValueError('Invalid learned weights')
            return bank
        obj = cls()
        flags = feature_tuple(data.get('active_features'))
        obj.regimes = {}
        for key, raw in (data.get('regimes') or {}).items():
            if len(key) != 3 or any(char not in '01' for char in key):
                raise ValueError('Invalid learning regime key')
            obj.regimes[key] = read_bank(raw)
        # The existing version-1 fields remain authoritative for the active bank.
        # Older files have no regime metadata and are therefore legacy (000).
        active = cls._regime_key(flags)
        obj.regimes[active] = read_bank(data)
        obj._bind(obj.regimes[active])
        obj.active_features = flags
        return obj

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def posterior(self, spot):
        raw = self.weights[spot]
        maximum = max(raw)
        weights = [math.exp(x - maximum) for x in raw]
        total = sum(weights)
        return [x / total for x in weights]

    def sample(self, spot, rng):
        return rng.choices(PARTICLES, weights=self.posterior(spot), k=1)[0]

    def confidence(self, spot):
        return self.counts[spot] / (self.counts[spot] + 20.)

    def learn(self, last, before_reps=None, forecast=None, rules=None, strength=1., features=None):
        if features is not None:
            self.activate(features)
        fair_choice, walk_over, _ = self.active_features
        rules = rules or Rules.read({})
        before_reps = before_reps or {}
        kinds = events(last.get('events'))
        weather = last.get('weather', 'clear')
        if isinstance(weather, dict):
            weather = weather.get('kind') or weather.get('type') or 'clear'
        for spot, info in spot_results(last).items():
            if spot not in SPOTS:
                continue
            customers = max(0., float(info.get('customers', 0)))
            records = info.get('trucks') or []
            trucks = [(float(t['price']), t['quality'],
                       float(before_reps.get(identity(t), {}).get(spot, 50)), int(t['meals']))
                      for t in records if t.get('quality') in QUALITIES]
            records = [t for t in records if t.get('quality') in QUALITIES]
            # Walk-over changes local arrivals after the origin crowd is drawn.
            # Until a joint customer-flow likelihood exists, avoid falsely fitting
            # those transferred sales as local preferences. Origin crowd residuals
            # below remain usable and segregated in this feature regime.
            if customers and trucks and not walk_over:
                losses = []
                for particle in PARTICLES:
                    predictions = expected_sales(trucks, customers, customer_model(spot, kinds, particle),
                                                 rules.values, fair_choice=True) if fair_choice else expected_sales(
                        trucks, customers, customer_model(spot, kinds, particle), rules.values)
                    loss = 0.
                    for predicted, t in zip(predictions, records):
                        actual = float(t.get('sold', 0))
                        # A sellout is a lower bound, never evidence that demand stops there.
                        error = max(0., actual - predicted) if actual >= t['meals'] else predicted - actual
                        loss += error * error / (actual + 12.)
                    losses.append(loss / max(1, len(records)))
                updated = [.98 * w - strength * min(loss, 80.) / 5.
                           for w, loss in zip(self.weights[spot], losses)]
                top = max(updated)
                self.weights[spot] = [max(-16., w - top) for w in updated]
                self.counts[spot] += 1
            if forecast:
                expected = (forecast.get('expected_customers') or {}).get(spot)
                if expected is not None and float(expected) > 0:
                    odds = weather_probs(forecast)
                    average = sum(p * weather_factor(w, spot) for w, p in odds.items())
                    conditional = float(expected) * weather_factor(str(weather), spot) / max(average, .1)
                    ratio = clamp(customers / max(conditional, 1.), .25, 3.)
                    key = str(weather)
                    old = self.crowd[spot].get(key, [1., 0])
                    # Shrink toward 1 when data are sparse. Event/weather effects in the
                    # supplied expected counts are never multiplied in a second time.
                    self.crowd[spot][key] = [.9 * old[0] + .1 * ratio, old[1] + 1]
                    # Online variance of weather-normalized forecast errors, including
                    # zero-customer days. Older checkpoints start with the prior.
                    count, mean, m2 = self.crowd_variance[spot].get(key, [0, 0., 0.])
                    observation = clamp(customers / max(conditional, 1.), 0., 3.)
                    count += 1
                    delta = observation - mean
                    mean += delta / count
                    m2 += delta * (observation - mean)
                    self.crowd_variance[spot][key] = [count, mean, m2]
        self.rows += 1

    def crowd_multiplier(self, spot, weather):
        mean, count = self.crowd[spot].get(weather, [1., 0])
        return 1. + (mean - 1.) * count / (count + 12.)

    def crowd_noise(self, spot, weather):
        """Relative standard deviation; shrink sparse observations toward old +/-15% noise."""
        count, mean, m2 = self.crowd_variance[spot].get(weather, [0, 0., 0.])
        prior_variance = .15**2 / 3.
        relative_m2 = m2 / max(.25, mean)**2 if count else 0.
        variance = (relative_m2 + 12 * prior_variance) / (max(0, count-1) + 12)
        return clamp(math.sqrt(variance), .04, .40)


def training_rows(path):
    """Saved collection pairs include exact pre-day local reputations and forecasts."""
    with open(path) as source:
        for line in source:
            row = json.loads(line)
            if 'pre' in row and 'last_day' in row:
                yield row


def train(rows):
    learner = Learner()
    for row in rows:
        reps = {identity(t): t.get('rep') or t.get('reputation') or {} for t in row['pre'].get('trucks', [])}
        learner.learn(row['last_day'], reps, row['pre'].get('forecast'),
                      features=row['pre'].get('features') or {})
    return learner
