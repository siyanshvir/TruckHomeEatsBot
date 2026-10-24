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


# Weather factors for spots registered at run time (new spots); the original six use WEATHER above.
SPOT_WEATHER = {}
_REGISTERED = []
# Mild default per weather for new spots; the Mirror is outdoors, the Med Center is rain-proof.
_NEW_SPOT_WEATHER = {
    'mirror': {'clear': 1., 'cloudy': .95, 'rain': .5, 'cold': .8},
    'medcenter': {'clear': 1., 'cloudy': 1., 'rain': 1., 'cold': 1.},
}
_MILD_WEATHER = {'clear': 1., 'cloudy': .97, 'rain': .9, 'cold': .95}


def known_spots():
    """The original six plus any spots registered at run time, in registration order."""
    return SPOTS + tuple(_REGISTERED)


def register_spots(priors, distances, weather=None):
    """Add new spots to PRIORS/DISTANCES/SPOT_WEATHER in place; all-or-nothing, original data untouched.

    priors: {spot: (mu, sd, mix, weight)} or {spot: {'mu','sd','mix','weight'}}.
    distances: {(a, b): metres} or {a: {b: metres}}; pairs between two original spots are ignored.
    weather: optional {spot: {weather: factor}}; default is the built-in new-spot table, else mild.
    """
    new_priors, new_weather, new_dist = {}, {}, {}
    for spot, raw in dict(priors or {}).items():
        if spot in SPOTS or not isinstance(spot, str):
            continue
        if isinstance(raw, dict):
            raw = (raw['mu'], raw['sd'], raw['mix'], raw['weight'])
        mu, sd, mix, weight = raw
        mix = tuple(float(x) for x in mix)
        values = [float(mu), float(sd), float(weight)] + list(mix)
        if len(mix) != 3 or not all(math.isfinite(x) for x in values) or sd <= 0:
            raise ValueError('Invalid prior for %s' % spot)
        new_priors[spot] = (float(mu), float(sd), mix, float(weight))
        table = dict(_NEW_SPOT_WEATHER.get(spot, _MILD_WEATHER))
        table.update(dict(weather or {}).get(spot) or {})
        new_weather[spot] = {str(k): float(v) for k, v in table.items()}
    allowed = set(SPOTS) | set(_REGISTERED) | set(new_priors)
    pairs = []
    for key, value in dict(distances or {}).items():
        if isinstance(value, dict):
            pairs.extend(((key, other), d) for other, d in value.items())
        else:
            pairs.append((key, value))
    for (left, right), distance in pairs:
        distance = float(distance)
        if left == right or left not in allowed or right not in allowed or not math.isfinite(distance) or distance < 0:
            continue
        if left in SPOTS and right in SPOTS:
            continue
        new_dist.setdefault(left, {})[right] = distance
        new_dist.setdefault(right, {})[left] = distance
    PRIORS.update(new_priors)
    SPOT_WEATHER.update(new_weather)
    for spot in new_priors:
        if spot not in _REGISTERED:
            _REGISTERED.append(spot)
    for spot, row in new_dist.items():
        DISTANCES.setdefault(spot, {}).update(row)
    for spot in new_priors:
        DISTANCES.setdefault(spot, {})


def reset_spots():
    """Drop every registered spot (tests and kill-switch); the original six are restored untouched."""
    for spot in _REGISTERED:
        PRIORS.pop(spot, None)
        SPOT_WEATHER.pop(spot, None)
        DISTANCES.pop(spot, None)
    for row in DISTANCES.values():
        for spot in _REGISTERED:
            row.pop(spot, None)
    del _REGISTERED[:]


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
    for name, words in (('game', ('game', 'football', 'championship')), ('midterm', ('midterm', 'exam')),
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


def without_null_counts(state):
    """The API can send null for a spot's expected customers. Treat that as missing, not zero."""
    forecast = state.get('forecast') or {}
    counts = forecast.get('expected_customers') or {}
    if all(v is not None for v in counts.values()):
        return state
    known = {spot: v for spot, v in counts.items() if v is not None}
    return dict(state, forecast=dict(forecast, expected_customers=known))


def weather_probs(forecast):
    raw = forecast.get('weather') or {'clear': 1.}
    if isinstance(raw, str):
        raw = {raw: 1.}
    probs = {str(k).lower(): max(0., float(v)) for k, v in raw.items()}
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items() if v > 0} if total else {'clear': 1.}


def weather_factor(weather, spot):
    if spot in SPOTS:
        return WEATHER.get(weather, WEATHER['clear'])[SPOTS.index(spot)]
    table = SPOT_WEATHER.get(spot) or {}
    return table.get(weather, table.get('clear', 1.))


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
                   dict(r.get('values') or VALUES),
                   tuple(s for s in (r.get('spots') or SPOTS) if s in SPOTS or s in _REGISTERED),
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


def expected_sales(trucks, customers, model, values=VALUES, fair_choice=False):
    """Sequential choice with finite stock; exact ties favor earlier trucks."""
    if fair_choice:
        return list(_local_flow(tuple(tuple(t) for t in trucks), max(0., float(customers)),
                                (model[0], model[1], tuple(model[2])),
                                tuple(sorted(values.items())), True)[0])
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


# Official campus distances in metres, symmetric. Walking is modeled as an
# expected fluid flow; the live engine's individual random draws are not replayed.
_DISTANCE_PAIRS = {
    ('oval', 'thompson'): 130, ('oval', 'union'): 470, ('oval', 'stadium'): 590,
    ('oval', 'highst'): 440, ('oval', 'northres'): 550,
    ('thompson', 'union'): 520, ('thompson', 'stadium'): 490,
    ('thompson', 'highst'): 580, ('thompson', 'northres'): 640,
    ('union', 'stadium'): 990, ('union', 'highst'): 570, ('union', 'northres'): 910,
    ('stadium', 'highst'): 980, ('stadium', 'northres'): 840,
    ('highst', 'northres'): 410,
}
DISTANCES = {spot: {} for spot in SPOTS}
for (_left, _right), _distance in _DISTANCE_PAIRS.items():
    DISTANCES[_left][_right] = DISTANCES[_right][_left] = _distance


WALK_DISTANCES = DISTANCES


@lru_cache(maxsize=32768)
def _scalar_budget_cdf(price, mu, sd):
    return .5 * (1. + math.erf((price-mu) / (max(sd, .3)*math.sqrt(2.))))


def _budget_cdf(price, model):
    return _scalar_budget_cdf(price, model[0], model[1])


@lru_cache(maxsize=4096)
def _budget_cohorts(model, prices):
    edges = (float('-inf'),) + tuple(sorted(set(prices))) + (float('inf'),)
    return tuple((mass * (_budget_cdf(upper, model)-_budget_cdf(lower, model)), minimum, lower, upper)
                 for minimum, mass in enumerate(model[2]) if mass > 0
                 for lower, upper in zip(edges, edges[1:])
                 if _budget_cdf(upper, model)-_budget_cdf(lower, model) > 1e-12)


def _choices(trucks, minimum, budget, scores):
    return tuple(i for i, truck in enumerate(trucks)
                 if truck[3] > 0 and QUALITIES.index(truck[1]) >= minimum
                 and truck[0] <= budget and scores[i] >= -1e-9)


def _shares(eligible, remaining, scores, fair_choice):
    available = [i for i in eligible if remaining[i] > 1e-8]
    if not available:
        return ()
    if len(available) == 1:
        return ((available[0], 1.),)
    if not fair_choice:
        best = max(available, key=lambda i: (scores[i], -i))
        return ((best, 1.),)
    top = max(scores[i] for i in available)
    weighted = [(i, math.exp((scores[i]-top)/1.5)) for i in available]
    total = sum(weight for _, weight in weighted)
    return tuple((i, weight/total) for i, weight in weighted)


@lru_cache(maxsize=8192)
def _local_flow(trucks, customers, model, value_items, fair_choice):
    values = dict(value_items)
    cohorts = _budget_cohorts(model, tuple(t[0] for t in trucks))
    scores = [reputation_value(q, r, values)-p for p, q, r, _ in trucks]
    eligible = [_choices(trucks, minimum, lower, scores) for _, minimum, lower, _ in cohorts]
    # Deterministic legacy ties additionally favor a lower selling price.
    if not fair_choice:
        eligible = [tuple(sorted(options, key=lambda i: (-scores[i], trucks[i][0], i))) for options in eligible]
    remaining = [max(0., float(t[3])) for t in trucks]
    sold, served = [0.]*len(trucks), [0.]*len(cohorts)
    duration = 1.
    for _ in range(len(trucks)+1):
        rates, cohort_rates = [0.]*len(trucks), [0.]*len(cohorts)
        for k, (mass, _, _, _) in enumerate(cohorts):
            options = eligible[k]
            if fair_choice:
                shares = _shares(options, remaining, scores, True)
            else:
                first = next((i for i in options if remaining[i] > 1e-8), None)
                shares = () if first is None else ((first, 1.),)
            for i, weight in shares:
                rate = mass*customers*weight
                rates[i] += rate
                cohort_rates[k] += rate
        dt = min([duration]+[remaining[i]/rate for i, rate in enumerate(rates) if rate > 0 and remaining[i] > 1e-8])
        for i, rate in enumerate(rates):
            bought = min(remaining[i], rate*dt)
            remaining[i] -= bought
            sold[i] += bought
        for k, rate in enumerate(cohort_rates):
            served[k] += rate*dt
        duration -= dt
        if duration <= 1e-8 or not any(rates):
            break
    residual = tuple((max(0., mass*customers-served[k]), minimum, lower, upper)
                     for k, (mass, minimum, lower, upper) in enumerate(cohorts)
                     if mass*customers-served[k] > 1e-9)
    return tuple(sold), residual


def expected_market_sales(placements, crowds, models, values=VALUES, fair_choice=False, walk_over=False):
    """Expected sales aligned with each spot's truck list.

    First serve all original local crowds. If enabled, original hungry budget and
    quality cohorts may walk once to their nearest acceptable other spot, with
    probability .7*exp(-distance/400). Remaining inventory is shared across origins.
    This fluid approximation preserves customers and stock, rather than resampling
    a new budget/quality or counting a customer at multiple destinations.
    """
    spots = tuple(dict.fromkeys(tuple(placements)+tuple(crowds)))
    if not walk_over:
        return {spot: expected_sales(placements.get(spot, []), crowds.get(spot, 0.), models[spot],
                                     values, fair_choice=fair_choice) for spot in spots}
    items = tuple(sorted(values.items()))
    sold, remaining, residual = {}, {}, {}
    for spot in spots:
        trucks = tuple(tuple(t) for t in placements.get(spot, []))
        model = models[spot]
        result, hungry = _local_flow(trucks, max(0., float(crowds.get(spot, 0.))),
                                    (model[0], model[1], tuple(model[2])), items, fair_choice)
        sold[spot] = list(result)
        remaining[spot] = [max(0., float(t[3])-result[i]) for i, t in enumerate(trucks)]
        residual[spot] = hungry
    # Split residual budget intervals at all destination prices; this retains
    # each origin's budget distribution after its own local selection.
    all_prices = sorted({t[0] for spot in spots for t in placements.get(spot, [])})
    cohorts = []
    scores = {spot: [reputation_value(t[1], t[2], values)-t[0] for t in placements.get(spot, [])]
              for spot in spots}
    open_spots = tuple(spot for spot in spots if any(stock > 1e-8 for stock in remaining[spot]))
    choices_cache = {}
    for origin in spots:
        model = models[origin]
        for mass, minimum, lower, upper in residual[origin]:
            edges = [lower]+[price for price in all_prices if lower < price < upper]+[upper]
            denominator = _budget_cdf(upper, model)-_budget_cdf(lower, model)
            for left, right in zip(edges, edges[1:]):
                quantity = mass*(_budget_cdf(right, model)-_budget_cdf(left, model))/max(denominator, 1e-15)
                if quantity < 1e-10:
                    continue
                destinations = []
                for destination in open_spots:
                    distance = DISTANCES.get(origin, {}).get(destination)
                    if distance is None:
                        continue
                    key = (destination, minimum, left)
                    options = choices_cache.get(key)
                    if options is None:
                        options = tuple(i for i in _choices(placements.get(destination, []), minimum, left, scores[destination])
                                        if remaining[destination][i] > 1e-8)
                        if not fair_choice:
                            options = tuple(sorted(options, key=lambda i: (-scores[destination][i], placements[destination][i][0], i)))
                        choices_cache[key] = options
                    if options:
                        destinations.append((distance, destination, options))
                if destinations:
                    cohorts.append((quantity, tuple(sorted(destinations))))
    duration = 1.
    for _ in range(sum(len(remaining[spot]) for spot in spots)+1):
        rates = {spot: [0.]*len(remaining[spot]) for spot in spots}
        for quantity, destinations in cohorts:
            for distance, destination, eligible in destinations:
                if fair_choice:
                    shares = _shares(eligible, remaining[destination], scores[destination], True)
                else:
                    first = next((i for i in eligible if remaining[destination][i] > 1e-8), None)
                    shares = () if first is None else ((first, 1.),)
                if shares:
                    flow = quantity*.7*math.exp(-distance/400.)
                    for i, weight in shares:
                        rates[destination][i] += flow*weight
                    break
        if not any(rate > 0 for local in rates.values() for rate in local):
            break
        dt = min([duration]+[remaining[spot][i]/rate for spot in spots for i, rate in enumerate(rates[spot])
                             if rate > 0 and remaining[spot][i] > 1e-8])
        for spot in spots:
            for i, rate in enumerate(rates[spot]):
                bought = min(remaining[spot][i], rate*dt)
                remaining[spot][i] -= bought
                sold[spot][i] += bought
        duration -= dt
        if duration <= 1e-8:
            break
    return sold
