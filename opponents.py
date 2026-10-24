"""Online probabilistic rival forecasts, scored before observing each new result.

A mixture learns which behavior models explain each rival. Contextual historical
neighbors use announced events, forecast crowd shares, and weather probabilities.
Price Wars' documented undercut/cycle behaviors are pricing hypotheses, not rules
that our own order must obey. No future or private orders are used.
"""
import math
from pricing_models import reference_prices
from game import SPOTS, QUALITIES, COSTS, VALUES, clamp, events, weather_probs

LOCATION_MODELS = ('uniform', 'habit', 'repeat', 'crowd', 'context', 'opportunity', 'follow')
PRICE_MODELS = ('context', 'repeat', 'undercut', 'cycle', 'follow')
# A new behavior starts well below the incumbents (log weight), and must earn its share.
NEW_EXPERT_LOG_WEIGHT = -2.
FOLLOW_DEPTH_PRIOR, FOLLOW_DEPTH_RANGE = 1., (-1., 6.)


def normalize(values):
    total = sum(values)
    return [x/total for x in values] if total else [1/len(values)]*len(values)


def posterior(logs):
    maximum = max(logs)
    return normalize([math.exp(x-maximum) for x in logs])


def context(forecast):
    forecast = forecast or {}
    counts = forecast.get('expected_customers') or {}
    total = sum(float(v) for v in counts.values()) or 1.
    return {'events': events(forecast.get('events')), 'weather': weather_probs(forecast),
            'shares': {s: float(counts.get(s, 0))/total for s in SPOTS}}


def similarity(left, right):
    le, re = set(left.get('events', [])), set(right.get('events', []))
    event_distance = len(le ^ re)
    crowd_distance = sum(abs(left.get('shares', {}).get(s, 0)-right.get('shares', {}).get(s, 0)) for s in SPOTS)
    weather_distance = sum(abs(left.get('weather', {}).get(w, 0)-right.get('weather', {}).get(w, 0))
                           for w in ('clear', 'cloudy', 'rain', 'cold'))
    return math.exp(-1.5*event_distance-2.*crowd_distance-.5*weather_distance)


class OpponentModel:
    def __init__(self, data=None):
        self.data = data or {}
        self.ours = []  # our own recent records, oldest first: what a rival that follows us has seen

    def entry(self, name):
        e = self.data.setdefault(name, {'location': [0.]*(len(LOCATION_MODELS)-1) + [NEW_EXPERT_LOG_WEIGHT],
                                       'price': [0.]*(len(PRICE_MODELS)-1) + [NEW_EXPERT_LOG_WEIGHT],
                                       'active': 0., 'absent': 0., 'scored': 0,
                                       'log_loss': 0., 'repeat_log_loss': 0.,
                                       'price_count': 0, 'price_error': 0., 'repeat_price_error': 0.})
        # Checkpoints saved before the follow behavior existed keep their weights.
        for key, names in (('location', LOCATION_MODELS), ('price', PRICE_MODELS)):
            e[key] = list(e[key]) + [NEW_EXPERT_LOG_WEIGHT]*(len(names)-len(e[key]))
        e.setdefault('follow_depth', FOLLOW_DEPTH_PRIOR)
        return e

    def attendance(self, name):
        e = self.entry(name)
        # Unknown trucks get an explicitly uncertain prior; a single absence is
        # weak evidence. Multi-day absences reduce both competition and demand.
        return clamp((3.+e['active'])/(4.+e['active']+e['absent']), .05, .98)

    def weights(self, history, forecast):
        current = context(forecast)
        return [.96**(len(history)-index-1) * (.05+.95*similarity(r.get('context', {}), current))
                for index, r in enumerate(history)]

    def location_experts(self, name, history, forecast, histories, spots=SPOTS):
        counts = (forecast or {}).get('expected_customers') or {}
        uniform = [1/len(spots)]*len(spots)
        habit, nearby = [.2]*len(spots), [.2]*len(spots)
        weights = self.weights(history, forecast)
        for age, (rec, weight) in enumerate(zip(history, weights)):
            if rec['spot'] in spots:
                i = spots.index(rec['spot'])
                habit[i] += .96**(len(history)-age-1)
                nearby[i] += weight*3.
        repeat = [.2/len(spots)]*len(spots)
        if history and history[-1]['spot'] in spots:
            repeat[spots.index(history[-1]['spot'])] += .8
        crowd = normalize([max(0., float(counts.get(s, 0)))**3 for s in spots])
        pressure = {s: 0. for s in spots}
        for rival, records in histories.items():
            if rival == name or not records:
                continue
            total = sum(.96**age for age in range(len(records)))
            for age, rec in enumerate(reversed(records)):
                if rec['spot'] in pressure:
                    pressure[rec['spot']] += rec['meals']*.96**age/total
        opportunity = normalize([float(counts.get(s, 0))/(1.+pressure[s]/max(1., float(counts.get(s, 0)))) for s in spots])
        follow = [.1/len(spots)]*len(spots)  # mass where we have been recently, newest most
        for age, rec in enumerate(reversed(self.ours[-6:])):
            if rec.get('spot') in spots:
                follow[spots.index(rec['spot'])] += .6**age
        return [uniform, normalize(habit), normalize(repeat), crowd, normalize(nearby), opportunity,
                normalize(follow)]

    def placement(self, name, history, forecast, histories, spots=SPOTS):
        experts = self.location_experts(name, history, forecast, histories, spots)
        weights = posterior(self.entry(name)['location'])
        # A uniform floor keeps unobserved moves possible as rivals change code.
        return normalize([.05/len(spots)+.95*sum(w*p[i] for w, p in zip(weights, experts)) for i in range(len(spots))])

    def quality(self, history, forecast):
        weights = dict.fromkeys(QUALITIES, .3)
        for r, w in zip(history, self.weights(history, forecast)):
            weights[r['quality']] += w*3.
        return normalize([weights[q] for q in QUALITIES])

    def price_experts(self, name, history, forecast, histories, quality, spot=None):
        paired = [(r, w) for r, w in zip(history, self.weights(history, forecast)) if r['quality'] == quality]
        default = {'cheap': 6.5, 'standard': 9., 'premium': 10.}[quality]
        if paired:
            total = sum(w for _, w in paired)
            mean = sum(r['price']*w for r, w in paired)/total
            variance = sum(w*(r['price']-mean)**2 for r, w in paired)/total
            stock = sum(r['meals']*w for r, w in paired)/total
            repeat = paired[-1][0]['price']
        else:
            mean, variance, stock, repeat = default, 1., 80., default
        competitors = [h[-1]['price'] for rival, h in histories.items() if rival != name and h
                       and h[-1]['quality'] == quality and (spot is None or h[-1]['spot'] == spot)]
        floor, upper = COSTS[quality]+.5, min(VALUES[quality]-.5, default+3.)
        undercut, cycle = reference_prices(competitors, mean, floor, upper)
        # Follow: undercut the price our truck charged last, by the depth this rival has shown.
        seen = [r for r in self.ours[-3:] if r.get('quality') == quality and (spot is None or r.get('spot') == spot)]
        follow = clamp(float(seen[-1]['price']) - self.entry(name)['follow_depth'], floor, upper) if seen else mean
        return [mean, repeat, undercut, cycle, follow], max(.2, min(2., math.sqrt(variance+.16))), stock

    def pricing(self, name, history, forecast, histories, quality, spot=None):
        experts, spread, stock = self.price_experts(name, history, forecast, histories, quality, spot)
        weights = posterior(self.entry(name)['price'])
        mean = sum(w*x for w, x in zip(weights, experts))
        variance = sum(w*(x-mean)**2 for w, x in zip(weights, experts))
        return mean, min(2.5, math.sqrt(spread*spread+variance)), stock

    def observe(self, name, record, history, forecast, histories, spots=SPOTS):
        e = self.entry(name)
        # Gently forget old attendance so a previously inactive truck can return.
        e['active'] *= .97
        e['absent'] *= .97
        e['active' if record else 'absent'] += 1.
        if record is None:
            return
        if forecast and (forecast.get('expected_customers') or {}):
            experts = self.location_experts(name, history, forecast, histories, spots)
            probabilities = self.placement(name, history, forecast, histories, spots)
            index = spots.index(record['spot'])
            e['log_loss'] -= math.log(max(probabilities[index], 1e-9))
            e['repeat_log_loss'] -= math.log(max(experts[2][index], 1e-9))
            e['scored'] += 1
            updated = [.98*w+.7*math.log(max(p[index], 1e-6)) for w, p in zip(e['location'], experts)]
            maximum = max(updated)
            e['location'] = [max(-12., x-maximum) for x in updated]
        if self.ours and record['spot'] == self.ours[-1].get('spot') and record['quality'] == self.ours[-1].get('quality'):
            low, high = FOLLOW_DEPTH_RANGE
            depth = clamp(float(self.ours[-1]['price']) - float(record['price']), low, high)
            e['follow_depth'] = .5*e['follow_depth'] + .5*depth
        if history:
            quality = record['quality']
            experts, spread, _ = self.price_experts(name, history, forecast, histories, quality, record['spot'])
            predicted, _, _ = self.pricing(name, history, forecast, histories, quality, record['spot'])
            e['price_count'] += 1
            e['price_error'] += abs(predicted-record['price'])
            e['repeat_price_error'] += abs(experts[1]-record['price'])
            updated = [.98*w-.35*min(25., ((p-record['price'])/max(.8, spread))**2)
                       for w, p in zip(e['price'], experts)]
            maximum = max(updated)
            e['price'] = [max(-12., x-maximum) for x in updated]

    def summary(self, name, history, forecast, histories, spots=SPOTS):
        e = self.entry(name)
        return {'attendance': self.attendance(name), 'history_days': len(history),
                'locations': dict(zip(spots, self.placement(name, history, forecast, histories, spots))),
                'location_models': dict(zip(LOCATION_MODELS, posterior(e['location']))),
                'price_models': dict(zip(PRICE_MODELS, posterior(e['price'])))}
