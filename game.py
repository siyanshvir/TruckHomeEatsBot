"""Published game contracts and an expected customer-flow model. No network or learning."""
import math
from functools import lru_cache
from dataclasses import dataclass

SPOTS = ('highst', 'oval', 'union', 'northres', 'thompson', 'stadium')
QUALITIES = ('cheap', 'standard', 'premium')
COSTS = dict(zip(QUALITIES, (2.5, 3.75, 5.5)))
VALUES = dict(zip(QUALITIES, (8., 12., 16.)))
REP_CHANGE = dict(zip(QUALITIES, (-3, 2, 5)))
# Priors are hypotheses, not engine constants. Historical observations update them.
PRIORS = {
    'highst': (12., 4., (.50, .35, .15), .24),
    'oval': (13., 3., (.40, .40, .20), .20),
    'union': (13., 3., (.20, .60, .20), .18),
    'northres': (11., 2.5, (.50, .42, .08), .16),
    'thompson': (12., 3., (.47, .40, .13), .14),
    'stadium': (13., 3., (.25, .50, .25), .08),
}
WEATHER = {
    'clear': (1., 1., 1., 1., 1., 1.),
    'cloudy': (.95, 1., .95, .94, .96, .98),
    'rain': (.71, .53, .90, .85, .90, .55),
    'cold': (.85, .79, .98, .90, .98, .78),
}


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def identity(truck):
    return str(truck.get('id') or truck.get('truck_name') or truck.get('name') or '?')


def events(raw):
    if isinstance(raw, dict):
        raw = [dict(v, key=k) if isinstance(v, dict) else k for k, v in raw.items()]
    if isinstance(raw, str):
        raw = [raw]
    text = ' '.join(str(x) for x in (raw or [])).lower()
    out = []
    for name, words in (('game', ('game', 'football')), ('midterm', ('midterm', 'exam')),
                        ('concert', ('concert',)), ('late', ('late', 'rush')),
                        ('rally', ('rally',))):
        if any(w in text for w in words):
            out.append(name)
    return out


def customer_model(spot, kinds, particle):
    mu, sd, mix, _ = PRIORS[spot]
    if 'game' in kinds:
        mu += 1. if spot == 'stadium' else .5
        if spot == 'stadium':
            mix = (.45, .40, .15)
    if 'midterm' in kinds and spot == 'thompson':
        mu += .5
        mix = (.10, .35, .55)
    if 'concert' in kinds and spot == 'union':
        mu += 1.
    if 'late' in kinds and spot in ('highst', 'northres'):
        mix = (.75, .20, .05)
    shift, spread, taste = particle
    move = clamp(taste, -mix[2], mix[0])
    return mu + shift, sd * spread, (mix[0] - move, mix[1], mix[2] + move)


def weather_probs(forecast):
    raw = forecast.get('weather') or {'clear': 1.}
    if isinstance(raw, str):
        raw = {raw: 1.}
    probs = {str(k).lower(): max(0., float(v)) for k, v in raw.items()}
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items() if v > 0} if total else {'clear': 1.}


def weather_factor(weather, spot):
    return WEATHER.get(weather, WEATHER['clear'])[SPOTS.index(spot)]


def spot_results(last):
    raw = last.get('spots') or {}
    if isinstance(raw, list):
        return {x.get('spot') or x.get('id'): x for x in raw}
    return raw


@dataclass(frozen=True)
class Rules:
    budget: float
    costs: dict
    values: dict
    spots: tuple
    price_min: float
    price_max: float

    @classmethod
    def read(cls, state):
        r = state.get('rules') or {}
        return cls(float(r.get('budget', 500)), dict(r.get('costs') or COSTS),
                   dict(r.get('values') or VALUES), tuple(r.get('spots') or SPOTS),
                   float(r.get('price_min', 1)), float(r.get('price_max', 30)))

    def capacity(self, quality):
        return int((self.budget + 1e-8) / self.costs[quality])

    def validate(self, order):
        assert order['spot'] in self.spots
        assert order['quality'] in QUALITIES
        assert self.price_min <= order['price'] <= self.price_max
        assert isinstance(order['meals'], int) and 0 <= order['meals'] <= self.capacity(order['quality'])
        return order


def reputation_value(quality, reputation, values=VALUES):
    return values[quality] * (.7 + .006 * clamp(reputation, 0, 100))


@lru_cache(maxsize=4096)
def _customer_segments(trucks, model, value_items):
    """Cache crowd-independent preferences; retain the existing sequential flow model."""
    values = dict(value_items)
    mu, sd, mix = model
    scores = [reputation_value(q, r, values) - p for p, q, r, _ in trucks]
    segments = []
    for minimum, mass in enumerate(mix):
        eligible = [i for i, (p, q, _, stock) in enumerate(trucks)
                    if QUALITIES.index(q) >= minimum and scores[i] >= -1e-9 and stock > 0]
        prices = sorted({trucks[i][0] for i in eligible})
        for j, price in enumerate(prices):
            low = .5 * (1 + math.erf((price - mu) / (max(sd, .3) * math.sqrt(2))))
            high = (.5 * (1 + math.erf((prices[j+1] - mu) / (max(sd, .3) * math.sqrt(2))))
                    if j+1 < len(prices) else 1.)
            preference = sorted((i for i in eligible if trucks[i][0] <= price),
                                key=lambda i: (-scores[i], trucks[i][0], i))
            if high - low > 1e-10:
                segments.append((mass * (high - low), tuple(preference)))
    return tuple(segments)


def expected_sales(trucks, customers, model, values=VALUES):
    """Sequential choice with finite stock; exact ties favor earlier trucks."""
    mu, sd, mix = model
    segments = _customer_segments(tuple(tuple(t) for t in trucks),
                                  (mu, sd, tuple(mix)), tuple(sorted(values.items())))
    remaining = [float(t[3]) for t in trucks]
    sold = [0.] * len(trucks)
    duration = 1.
    for _ in range(len(trucks) + 1):
        rates = [0.] * len(trucks)
        for mass, preference in segments:
            for i in preference:
                if remaining[i] > 1e-8:
                    rates[i] += mass * customers
                    break
        dt = min([duration] + [remaining[i] / rate for i, rate in enumerate(rates)
                                if rate > 0 and remaining[i] > 1e-8])
        for i, rate in enumerate(rates):
            bought = min(remaining[i], rate * dt)
            remaining[i] -= bought
            sold[i] += bought
        duration -= dt
        if duration <= 1e-8 or not any(rates):
            break
    return sold
