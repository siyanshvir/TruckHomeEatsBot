"""Learned sales forecasting plus exhaustive location/quality profit optimization.

Every spot receives the same fixed search effort. No crowd-ordered early exit,
forced rival undercut, premium-only switch, or arbitrary selling-price cap.
"""
import hashlib
import math
import random
import time
from game import (SPOTS, QUALITIES, PRIORS, COSTS, REP_CHANGE, Rules, clamp,
                  customer_model, events, expected_sales, identity, reputation_value,
                  spot_results, weather_factor, weather_probs)
from learning import Learner
from opponents import OpponentModel, context


class Brain:
    def __init__(self, learner=None, seed=42, samples=12, risk=.15, price_cap=None, quiet_bonus=0., small_crowds=False):
        self.learner = learner or Learner()
        self.seed, self.samples, self.risk = seed, max(4, int(samples)), float(risk)
        # Optional steering: never charge above price_cap; reward spots where rivals rarely park.
        self.price_cap, self.quiet_bonus = price_cap, float(quiet_bonus)
        self.small_crowds = bool(small_crowds)
        self.low_price_caps = {'cheap': 5.50, 'standard': 7.00, 'premium': 8.50}
        self.market = None
        self.seen = None
        self.history = {}
        self.reputations = {}
        self.pending = None
        self.last_order = None
        self.diagnostics = {}
        self.opponents = OpponentModel()

    def dump(self):
        return {'learner': self.learner.to_dict(), 'market': self.market, 'seen': self.seen,
                'history': self.history, 'reputations': self.reputations,
                'pending': self.pending, 'last_order': self.last_order, 'opponents': self.opponents.data}

    def restore(self, data):
        self.learner = Learner.from_dict(data['learner'])
        self.opponents = OpponentModel(data.get('opponents'))
        for key in ('market', 'seen', 'history', 'reputations', 'pending', 'last_order'):
            setattr(self, key, data[key])

    def observe(self, state):
        key = [state.get('mode'), state.get('market')]
        if key != self.market:
            self.market = key
            self.seen = None
            self.reputations = {}
            self.pending = None
            self.last_order = None
        last = state.get('last_day') or {}
        if last.get('day') is None or (self.seen is not None and last['day'] <= self.seen):
            return
        me = identity(state.get('you') or {})
        rules = Rules.read(state)
        before = {name: dict(reps) for name, reps in self.reputations.items()}
        # Our local reputation is public; undo this day's increment to fit pre-day sales.
        own_reps = dict((state.get('you') or {}).get('reputation') or {})
        own = last.get('you') or {}
        if own.get('sold', 0) > 0 and own.get('spot') in own_reps:
            s = own['spot']
            # At the cap the exact previous value is ambiguous; prefer tracked history.
            own_reps[s] = before.get(me, {}).get(s, clamp(own_reps[s] - REP_CHANGE[own['quality']], 0, 100))
        before[me] = own_reps
        forecast = self.pending['forecast'] if self.pending and self.pending['day'] == last['day'] else None
        self.learner.learn(last, before, forecast, rules, strength=.7)
        observed_context = context(forecast or {'events': last.get('events') or []})
        visited = {}
        records = {}
        for spot, info in spot_results(last).items():
            if spot not in SPOTS:
                continue
            for truck in info.get('trucks') or []:
                name = identity(truck)
                visited[name] = (spot, truck)
                if truck.get('quality') in QUALITIES:
                    rec = {'spot': spot, 'price': float(truck['price']), 'quality': truck['quality'],
                           'meals': int(truck['meals']), 'sold': int(truck.get('sold', 0)),
                           'context': observed_context, 'day': last['day']}
                    records[name] = rec
        # Score each prediction before adding ANY of today's orders to history.
        # This prevents both future-data leakage and truck iteration-order effects.
        roster = {identity(t) for t in state.get('trucks') or []} | set(records)
        for name in sorted(roster - {me}):
            self.opponents.observe(name, records.get(name), self.history.get(name, []),
                                   forecast, self.history, rules.spots)
        for name, rec in records.items():
            self.history.setdefault(name, []).append(rec)
            self.history[name] = self.history[name][-48:]
        for name in set(self.reputations) | set(visited):
            reps = self.reputations.setdefault(name, {s: 50. for s in SPOTS})
            for spot in SPOTS:
                if name in visited and visited[name][0] == spot:
                    t = visited[name][1]
                    if t.get('sold', 0) > 0:
                        reps[spot] = clamp(reps[spot] + REP_CHANGE.get(t.get('quality'), 0), 0, 100)
                else:
                    reps[spot] += .03 * (50 - reps[spot])
        for t in state.get('trucks') or []:
            if isinstance(t.get('reputation'), dict):
                self.reputations[identity(t)] = dict(t['reputation'])
        self.reputations[me] = dict((state.get('you') or {}).get('reputation') or {s: 50. for s in SPOTS})
        self.seen = last['day']

    def forecast(self, state, rules):
        fc = state.get('forecast') or {}
        given = fc.get('expected_customers') or {}
        kinds = events(fc.get('events'))
        odds = weather_probs(fc)
        counts = {}
        for s in rules.spots:
            if s in given:  # Zero is meaningful; do not replace it with a prior.
                counts[s] = max(0., float(given[s]))
                continue
            base = 110 * max(4, len(state.get('trucks') or [])) * PRIORS[s][3]
            for event in kinds:
                mult = {'game': {'stadium': 5, 'highst': 1.5, '*': 1.2},
                        'midterm': {'thompson': 1.7}, 'concert': {'union': 1.8},
                        'late': {'highst': 2, 'northres': 2, '*': .5}, 'rally': {'oval': 2}}[event]
                base *= mult.get(s, mult.get('*', 1.))
            counts[s] = base * sum(p * weather_factor(w, s) for w, p in odds.items())
        return counts

    def scenarios(self, state, rules, counts, rng):
        """Joint rival placements and customer uncertainty, shared across candidate orders."""
        me = identity(state.get('you') or {})
        roster = [identity(t) for t in state.get('trucks') or [] if identity(t) != me]
        fc = state.get('forecast') or {}
        odds = weather_probs(fc)
        kinds = events(fc.get('events'))
        field = [h[-1] for name, h in self.history.items() if name in roster and h]
        scenarios = []
        for index in range(self.samples):
            # Stratified weather sampling reduces random ranking fluctuations.
            target, acc = (index + .5) / self.samples, 0.
            weather = list(odds)[-1]
            for w, p in sorted(odds.items()):
                acc += p
                if target <= acc:
                    weather = w
                    break
            placements = {s: [] for s in rules.spots}
            active = 0.
            for name in roster:
                hist = self.history.get(name, [])
                if rng.random() > self.opponents.attendance(name):
                    continue
                weights = self.opponents.placement(name, hist, fc, self.history, rules.spots)
                spot = rng.choices(rules.spots, weights=weights, k=1)[0]
                if hist:
                    quality_weights = self.opponents.quality(hist, fc)
                    q = rng.choices(QUALITIES, weights=quality_weights, k=1)[0]
                    mean, spread, stock = self.opponents.pricing(name, hist, fc, self.history, q, spot)
                    price = rng.gauss(mean, spread)
                    meals = max(0, min(rules.capacity(q), int(stock * rng.uniform(.85, 1.15))))
                else:
                    rec = rng.choice(field) if field else None
                    q = rec['quality'] if rec else rng.choice(QUALITIES)
                    price = rec['price'] if rec else {'cheap': 6.5, 'standard': 9., 'premium': 10.}[q]
                    price += rng.gauss(0, .7)
                    meals = min(rules.capacity(q), int(rec['meals']) if rec else 80)
                price = round(clamp(price, rules.price_min, rules.price_max), 2)
                rep = self.reputations.get(name, {}).get(spot, 50.)
                rep = clamp(rep + rng.gauss(0, 5 if hist else 12), 0, 100)
                placements[spot].append((price, q, rep, meals))
                active += min(1., meals * rules.costs[q] / 250.)
            crowds, models = {}, {}
            for spot in rules.spots:
                average = sum(p * weather_factor(w, spot) for w, p in odds.items())
                conditional = weather_factor(weather, spot) / max(average, .1)
                correction = self.learner.crowd_multiplier(spot, weather)
                crowds[spot] = counts[spot] * conditional * correction * rng.uniform(.85, 1.15)
                models[spot] = customer_model(spot, kinds, self.learner.sample(spot, rng))
            scenarios.append((placements, crowds, models, active))
        return scenarios

    def evaluate(self, state, rules, spot, quality, price, scenarios):
        rep = float((state.get('you') or {}).get('reputation', {}).get(spot, 50))
        cost = rules.costs[quality]
        cap = rules.capacity(quality)
        baseline = max(4, len(state.get('trucks') or []))
        distributions = []
        for placement, crowds, models, active in scenarios:
            trucks = placement[spot] + [(price, quality, rep, float('inf'))]
            # Estimate demand at three production levels, then interpolate while sizing
            # stock. This accounts for our own contribution to campus demand. The
            # four-truck floor often makes all three levels identical.
            levels = [max(4., active + contribution) / baseline for contribution in (0., .5, 1.)]
            demands = {}
            for scale in set(levels):
                demands[scale] = expected_sales(trucks, crowds[spot] * scale, models[spot], rules.values)[-1]
            distributions.append([demands[scale] for scale in levels])
        remaining = max(0, int(state.get('days', 30)) - int(state.get('order_for_day', 1)))
        rep_delta = clamp(rep + REP_CHANGE[quality], 0, 100) - rep
        best = None
        n = len(distributions)
        quiet = sum(1 for placement, _, _, _ in scenarios if not placement[spot]) / max(1, n)
        # Every feasible integer inventory is considered, including zero.
        for meals in range(cap + 1):
            contribution = min(1., meals * cost / 250.)
            left = 0 if contribution <= .5 else 1
            fraction = contribution * 2 if left == 0 else (contribution - .5) * 2
            demand = [d[left] * (1-fraction) + d[left+1] * fraction for d in distributions]
            sales = [min(meals, x) for x in demand]
            mean = sum(sales) / n
            profit = mean * price - meals * cost
            loss = sum(max(0., meals * cost - x * price) for x in sales) / n
            # Small, bounded future benefit: no incentive to lose hundreds buying reputation.
            sale_probability = sum(x > .1 for x in sales) / n
            future = .006 * rules.values[quality] * rep_delta * mean * min(remaining, 6) * .15 * sale_probability
            future = clamp(future, -25., 25.)
            exploration = 3. * min(1., remaining / 15.) * (1-self.learner.confidence(spot)) * sale_probability
            score = profit - self.risk * loss + future + exploration
            if meals and self.quiet_bonus:
                score += self.quiet_bonus * quiet
            if best is None or score > best['score']:
                best = {'score': score, 'profit': profit, 'sales': mean, 'loss': loss,
                        'demand': sum(demand) / n,
                        'expected_unsold': meals - mean,
                        'sellout_probability': sum(x >= meals for x in demand) / n if meals else 0.,
                        'order': {'spot': spot, 'quality': quality, 'price': price, 'meals': meals}}
        return best

    def location_restriction(self, results, counts, spots):
        if not self.small_crowds:
            return {'best': max(results, key=lambda r: r['score']), 'enabled': False}
        # User rule: the THREE SMALLEST CUSTOMER forecasts, not the lowest rival
        # occupancy. Prefer the bottom two; the third must clear a profit hurdle.
        ordered = sorted(spots, key=lambda s: (counts[s], s))
        eligible, preferred = ordered[:3], ordered[:2]
        allowed = [r for r in results if r['order']['spot'] in eligible]
        first_two = [r for r in allowed if r['order']['spot'] in preferred]
        best = max(first_two or allowed, key=lambda r: r['score'])
        third = [r for r in allowed if r['order']['spot'] not in preferred]
        hurdle = max(15., .15*max(0., best['profit']))
        if third:
            challenger = max(third, key=lambda r: r['score'])
            if challenger['profit'] > best['profit']+hurdle and challenger['score'] > best['score']:
                best = challenger
        return {'best': best, 'enabled': True, 'rank_by': 'forecast_customers',
                'eligible': eligible, 'preferred': preferred, 'third_profit_hurdle': hurdle,
                'price_caps': dict(self.low_price_caps)}

    def decide(self, state):
        started = time.perf_counter()
        self.observe(state)
        if state.get('order_for_day') is None:
            next_day = (state.get('forecast') or {}).get('day') or int(state.get('day', 0))+1
            state = dict(state, order_for_day=next_day)
        rules = Rules.read(state)
        counts = self.forecast(state, rules)
        digest = hashlib.sha256(repr((self.seed, self.market, state.get('order_for_day'))).encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], 'big'))
        scenarios = self.scenarios(state, rules, counts, rng)
        results, coverage = [], {}
        by_pair = {}
        for spot in rules.spots:
            rep = float((state.get('you') or {}).get('reputation', {}).get(spot, 50))
            coverage[spot] = []
            for quality in QUALITIES:
                coverage[spot].append(quality)
                ceiling = min(rules.price_max, self.price_cap or rules.price_max,
                              self.low_price_caps[quality] if self.small_crowds else rules.price_max,
                              reputation_value(quality, rep, rules.values))
                low = max(rules.price_min, rules.costs[quality] + .25)
                # Coarse prices plus value-equivalent rival thresholds; prices above
                # rivals remain eligible to capture overflow and reputation advantages.
                prices = {round(low + .75*k, 2) for k in range(int(max(0, ceiling-low) / .75)+1)}
                prices.add(round(ceiling, 2))
                thresholds = []
                for placements, _, _, _ in scenarios:
                    for rp, rq, rr, _ in placements[spot]:
                        thresholds.append(reputation_value(quality, rep, rules.values)
                                          - reputation_value(rq, rr, rules.values) + rp - .01)
                # Quantiles bound computation even in a large tournament field.
                thresholds.sort()
                if thresholds:
                    prices.update(round(thresholds[int((len(thresholds)-1)*q)], 2) for q in (0., .25, .5, .75, 1.))
                candidates = [self.evaluate(state, rules, spot, quality, price, scenarios)
                              for price in sorted(prices) if low <= price <= ceiling]
                if candidates:
                    result = max(candidates, key=lambda r: r['score'])
                    by_pair[spot, quality] = result
                    results.extend(candidates)
        # Same refinement allowance for each spot, after complete coarse coverage.
        for spot in rules.spots:
            local = [r for (s, _), r in by_pair.items() if s == spot]
            if not local:
                continue
            top = max(local, key=lambda r: r['score'])
            quality, base = top['order']['quality'], top['order']['price']
            rep = float((state.get('you') or {}).get('reputation', {}).get(spot, 50))
            ceiling = min(rules.price_max, self.price_cap or rules.price_max,
                              self.low_price_caps[quality] if self.small_crowds else rules.price_max,
                              reputation_value(quality, rep, rules.values))
            for step in (-.5, -.25, .25, .5):
                price = round(base + step, 2)
                if max(rules.price_min, rules.costs[quality]+.01) <= price <= ceiling:
                    results.append(self.evaluate(state, rules, spot, quality, price, scenarios))
        if not results:
            raise ValueError('No legal candidates for supplied rules')
        restriction = self.location_restriction(results, counts, rules.spots)
        best = restriction['best']
        order = rules.validate(best['order'])
        self.pending = {'day': state.get('order_for_day'), 'forecast': state.get('forecast') or {}}
        self.last_order = order
        ranked = []
        for spot in rules.spots:
            local = [r for r in results if r['order']['spot'] == spot]
            if local:
                r = max(local, key=lambda r: r['score'])
                ranked.append({**r, 'forecast': counts[spot]})
        self.diagnostics = {'elapsed': time.perf_counter()-started, 'coverage': coverage,
                            'candidates': len(results), 'ranking': sorted(ranked, key=lambda r: -r['score']),
                            'chosen': best, 'location_rule': {k: v for k, v in restriction.items() if k != 'best'},
                            'rivals': {identity(t): self.opponents.summary(identity(t),
                                self.history.get(identity(t), []), state.get('forecast') or {},
                                self.history, rules.spots) for t in state.get('trucks') or []
                                if identity(t) != identity(state.get('you') or {})}}
        return order
