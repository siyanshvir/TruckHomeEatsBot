"""Online Bayesian model averaging, shared by training and the playing policy."""
import json
import math
from pathlib import Path
from game import (SPOTS, QUALITIES, REP_CHANGE, PRIORS, Rules, clamp, customer_model,
                  events, expected_sales, identity, spot_results, weather_factor, weather_probs)

# Broad enough to represent uncertainty rather than fitting one guessed distribution.
PARTICLES = [(mu, sd, taste) for mu in (-3., -1.5, 0., 1.5, 3.)
             for sd in (.65, 1., 1.4) for taste in (-.12, 0., .12)]


class Learner:
    VERSION = 1

    def __init__(self):
        self.weights = {s: [0.] * len(PARTICLES) for s in SPOTS}
        self.counts = {s: 0 for s in SPOTS}
        self.crowd = {s: {} for s in SPOTS}
        self.crowd_variance = {s: {} for s in SPOTS}
        self.rows = 0

    def to_dict(self):
        return {'version': self.VERSION, 'weights': self.weights, 'counts': self.counts,
                'crowd': self.crowd, 'crowd_variance': self.crowd_variance, 'rows': self.rows}

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        if data.get('version') != cls.VERSION:
            raise ValueError('Unsupported learned-model version')
        for spot in SPOTS:
            weights = data['weights'][spot]
            if len(weights) != len(PARTICLES) or not all(math.isfinite(x) for x in weights):
                raise ValueError('Invalid learned weights')
        obj.weights = data['weights']
        obj.counts = data['counts']
        obj.crowd = data['crowd']
        obj.crowd_variance = data.get('crowd_variance', {s: {} for s in SPOTS})
        obj.rows = data.get('rows', 0)
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

    def learn(self, last, before_reps=None, forecast=None, rules=None, strength=1.):
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
            if customers and trucks:
                losses = []
                for particle in PARTICLES:
                    predictions = expected_sales(trucks, customers, customer_model(spot, kinds, particle), rules.values)
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
        learner.learn(row['last_day'], reps, row['pre'].get('forecast'))
    return learner
