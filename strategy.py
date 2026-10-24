"""Learned sales forecasting plus exhaustive location/quality profit optimization.

Every spot receives the same fixed search effort. No crowd-ordered early exit,
forced rival undercut, premium-only switch, or arbitrary selling-price cap.
"""
import hashlib
import math
import random
import time
from game import (SPOTS, QUALITIES, PRIORS, COSTS, REP_CHANGE, Rules, clamp,
                  customer_model, events, identity, reputation_value,
                  spot_results, weather_factor, weather_probs, without_null_counts)
from learning import Learner
from opponents import OpponentModel, context
from forecast_log import capture
from mechanics import features, demand_levels
from overheads import Overheads
from riots import riot_cost
from staffing import CrewModel

# Documented surprise rates; building spots can have a fire drill (open ones today).
BUILDINGS = ('thompson', 'union', 'northres')
FIRE_DRILL_CHANCE, GIVEAWAY_CHANCE = .06, .10
# Chance an unforeseen rival shows up at a spot: events and weight of a weak prior, then clamped.
PHANTOM_PRIOR, PHANTOM_LIMITS = (.64, 8.), (.03, .30)


class Brain:
    def __init__(self, learner=None, seed=42, samples=12, risk=.15, price_cap=None, quiet_bonus=0., small_crowds=False, spot_price_lift=None):
        self.learner = learner or Learner()
        self.seed, self.samples, self.risk = seed, max(4, int(samples)), float(risk)
        # Optional steering: never charge above price_cap; reward spots where rivals rarely park.
        self.price_cap, self.quiet_bonus = price_cap, float(quiet_bonus)
        self.small_crowds = bool(small_crowds)
        # Per-spot multiplier on the low price ceilings, e.g. {'thompson': 1.10}.
        self.spot_price_lift = dict(spot_price_lift or {})
        self.low_price_caps = {'cheap': 5.50, 'standard': 7.00, 'premium': 8.50}
        self.market = None
        self.seen = None
        self.history = {}
        self.reputations = {}
        self.pending = None
        self.last_order = None
        self.diagnostics = {}
        self.opponents = OpponentModel()
        self.crew = CrewModel()
        self.phantom = {'events': PHANTOM_PRIOR[0], 'exposures': PHANTOM_PRIOR[1]}
        self.prediction_checks = []
        self.forecast_record = None
        self.scenario_rivals = []

    def dump(self):
        return {'learner': self.learner.to_dict(), 'market': self.market, 'seen': self.seen,
                'history': self.history, 'reputations': self.reputations,
                'pending': self.pending, 'last_order': self.last_order, 'opponents': self.opponents.data,
                'prediction_checks': self.prediction_checks, 'crew': self.crew.to_dict(), 'phantom': self.phantom}

    def restore(self, data):
        self.learner = Learner.from_dict(data['learner'])
        self.opponents = OpponentModel(data.get('opponents'))
        self.prediction_checks = data.get('prediction_checks', [])
        self.crew = CrewModel(data.get('crew'))
        self.phantom = data.get('phantom') or self.bootstrap_phantom(self.prediction_checks)
        for key in ('market', 'seen', 'history', 'reputations', 'pending', 'last_order'):
            setattr(self, key, data[key])

    @staticmethod
    def bootstrap_phantom(checks):
        """Older checkpoints: learn the rate from the saved prediction checks."""
        phantom = {'events': PHANTOM_PRIOR[0], 'exposures': PHANTOM_PRIOR[1]}
        for check in checks:
            if check.get('order_matches') and not check.get('surprise'):
                phantom['exposures'] += 1
                phantom['events'] += min(1., max(0., check['actual_rivals'] - check['expected_rivals']))
        return phantom

    def game_day_rate(self):
        """Share of observed days with a game, shrunk toward 25%: how often game-day laws would bite."""
        days = {}
        for records in self.history.values():
            for record in records:
                days[record.get('day')] = days.get(record.get('day'), False) or (
                    'game' in ((record.get('context') or {}).get('events') or []))
        return (sum(days.values()) + 1.) / (len(days) + 4.)

    def phantom_rate(self):
        low, high = PHANTOM_LIMITS
        return min(high, max(low, self.phantom['events'] / self.phantom['exposures']))

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
        # state.features can describe tomorrow after a toggle. Fit completed
        # results only with their captured pre-order flags or explicit result flags.
        realized_features = last.get('features')
        if realized_features is None and self.pending and self.pending['day'] == last['day']:
            realized_features = self.pending.get('features')
        if realized_features is None and 'features' not in state:
            realized_features = {}  # Legacy archives and fixtures before feature flags.
        # Fire drills and giveaways thin a spot's crowd after the forecast; do not learn that as bias.
        surprise_spots = {m.get('spot') for m in last.get('moments') or []
                          if isinstance(m, dict) and m.get('kind') in ('firedrill', 'giveaway')}
        prediction = (self.pending or {}).get('prediction') if forecast else None
        if prediction:
            order = prediction['order']
            actual = spot_results(last).get(order['spot'], {})
            matched = all(own.get(k) == v for k, v in order.items())
            sold = float(own.get('sold', 0))
            check = {'mode': key[0], 'market': key[1], 'day': last['day'],
                     'forecast_id': prediction.get('forecast_id'),
                     'rules_digest': prediction.get('rules_digest'), 'regime_id': prediction.get('regime_id'),
                     'features': prediction.get('features'),
                     'order': order, 'order_matches': matched,
                     'expected_sales': prediction['sales'], 'actual_sales': sold,
                     'expected_profit': prediction['profit'], 'actual_profit': own.get('profit'),
                     'expected_rivals': prediction['rivals'],
                     'actual_rivals': sum(identity(t) != me for t in actual.get('trucks', [])),
                     'sellout': matched and order['meals'] > 0 and sold >= order['meals'],
                     'surprise': order['spot'] in surprise_spots}
            # A sold-out truck reveals only a lower bound on uncapped demand.
            check['demand_error'] = (max(0., sold-prediction['demand']) if check['sellout']
                                     else abs(sold-prediction['demand'])) if matched else None
            self.prediction_checks = (self.prediction_checks + [check])[-48:]
            if matched and not check['surprise']:
                self.phantom['exposures'] += 1
                self.phantom['events'] += min(1., max(0., check['actual_rivals'] - prediction['rivals']))
        if realized_features is not None:
            fit = last if not surprise_spots else dict(last, spots={s: v for s, v in spot_results(last).items()
                                                                    if s not in surprise_spots})
            self.learner.learn(fit, before, forecast, rules, strength=.7, features=realized_features)
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
                    rec['features'] = features(realized_features) if realized_features is not None else None
                    records[name] = rec
        # Score each prediction before adding ANY of today's orders to history.
        # This prevents both future-data leakage and truck iteration-order effects.
        roster = {identity(t) for t in state.get('trucks') or []} | set(records)
        self.opponents.ours = self.history.get(me, [])[-8:]  # before today's records are added
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
        try:
            self.crew.observe(state, me)
        except (TypeError, ValueError, KeyError, ZeroDivisionError):
            pass  # crew learning must never block ordering

    def crew_plan(self, state, order):
        """Headcount and wage for the accepted order; None when staff is not live."""
        if not (state.get('features') or {}).get('staff') or not isinstance(
                (state.get('you') or {}).get('staff'), dict):
            return None
        record = self.forecast_record if (self.forecast_record or {}).get('order') == order else None
        sales = [r['sales'] for r in (record or {}).get('scenarios', [])]
        return self.crew.plan(state, order, sales, identity(state.get('you') or {}))

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
        self.opponents.ours = self.history.get(me, [])[-8:]
        roster = [identity(t) for t in state.get('trucks') or [] if identity(t) != me]
        fc = state.get('forecast') or {}
        odds = weather_probs(fc)
        kinds = events(fc.get('events'))
        field = [h[-1] for name, h in self.history.items() if name in roster and h]
        scenarios = []
        self.scenario_rivals = []
        # Own stream: unforeseen rivals must not shift the draws used for everything else.
        phantoms = random.Random(hashlib.sha256(repr((self.seed, self.market, state.get('order_for_day'),
                                                      'phantom')).encode()).digest())
        rate = self.phantom_rate()
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
            named = []
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
                named.append({'id': name, 'spot': spot, 'price': price,
                              'quality': q, 'reputation': rep, 'meals': meals})
                active += min(1., meals * rules.costs[q] / 250.)
            thinned = {}
            if (state.get('features') or {}).get('surprises'):
                buildings = [s for s in rules.spots if s in BUILDINGS]
                if buildings and rng.random() < FIRE_DRILL_CHANCE:
                    thinned[rng.choice(buildings)] = .6
                if rng.random() < GIVEAWAY_CHANCE:
                    target = rng.choice(rules.spots)
                    thinned[target] = thinned.get(target, 1.) * rng.uniform(.6, .8)
            for spot in rules.spots:
                if field and phantoms.random() < rate:
                    rec = phantoms.choice(field)
                    q = rec['quality']
                    price = round(clamp(rec['price'] + phantoms.gauss(0, .7), rules.price_min, rules.price_max), 2)
                    rep = clamp(50. + phantoms.gauss(0, 12), 0, 100)
                    meals = min(rules.capacity(q), int(rec['meals']))
                    placements[spot].append((price, q, rep, meals))
                    named.append({'id': 'phantom', 'spot': spot, 'price': price, 'quality': q,
                                  'reputation': rep, 'meals': meals})
                    active += min(1., meals * rules.costs[q] / 250.)
            crowds, models = {}, {}
            for spot in rules.spots:
                average = sum(p * weather_factor(w, spot) for w, p in odds.items())
                conditional = weather_factor(weather, spot) / max(average, .1)
                correction = self.learner.crowd_multiplier(spot, weather)
                width = math.sqrt(3.) * self.learner.crowd_noise(spot, weather)
                crowds[spot] = (counts[spot] * conditional * correction * rng.uniform(1-width, 1+width)
                                * thinned.get(spot, 1.))
                models[spot] = customer_model(spot, kinds, self.learner.sample(spot, rng))
            scenarios.append((placements, crowds, models, active))
            self.scenario_rivals.append(named)
        return scenarios

    def evaluate(self, state, rules, spot, quality, price, scenarios):
        rep = float((state.get('you') or {}).get('reputation', {}).get(spot, 50))
        cost = rules.costs[quality]
        cap = rules.capacity(quality)
        flags = features(state.get('features'))
        overheads = Overheads(state, spot, flags['salvage'])
        if overheads.capacity is not None:
            cap = min(cap, overheads.capacity)  # each employee serves 50 meals a day
        charged = overheads.charged(price)
        riots = riot_cost(state, spot)  # score only: expected riot damage near tomorrow's game
        distributions = [demand_levels(state, rules, spot, quality, charged, scenario) for scenario in scenarios]
        remaining = max(0, int(state.get('days', 30)) - int(state.get('order_for_day', 1)))
        rep_delta = clamp(rep + REP_CHANGE[quality], 0, 100) - rep
        best = None
        inventory_options = []
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
            profit = overheads.profit(mean, meals, price, quality, cost)
            loss = sum(max(0., -overheads.profit(x, meals, price, quality, cost)) for x in sales) / n
            # Small, bounded future benefit: no incentive to lose hundreds buying reputation.
            sale_probability = sum(x > .1 for x in sales) / n
            future = .006 * rules.values[quality] * rep_delta * mean * min(remaining, 6) * .15 * sale_probability
            future = clamp(future, -25., 25.)
            exploration = 3. * min(1., remaining / 15.) * (1-self.learner.confidence(spot)) * sale_probability
            score = profit - self.risk * loss + future + exploration - riots
            if meals and self.quiet_bonus:
                score += self.quiet_bonus * quiet
            candidate = {'score': score, 'profit': profit, 'sales': mean, 'loss': loss,
                        'demand': sum(demand) / n,
                        'expected_unsold': meals - mean,
                        'expected_salvage': (meals-mean)*cost*(.25 if flags['salvage'] else 0.),
                        'sellout_probability': sum(x >= meals for x in demand) / n if meals else 0.,
                        'order': {'spot': spot, 'quality': quality, 'price': price, 'meals': meals}}
            inventory_options.append(candidate)
            if best is None or score > best['score']:
                best = candidate
        if self.small_crowds and best['order']['meals'] > 0 and best['profit'] > 0:
            base_meals = best['order']['meals']
            extra_limit = min(5, max(1, math.ceil(base_meals * .08)))
            tolerance = min(2., .03 * best['profit'])
            supported = []
            for option in inventory_options:
                extra = option['order']['meals'] - base_meals
                extra_sales = option['sales'] - best['sales']
                if (0 < extra <= extra_limit and extra_sales >= .60 * extra - 1e-9
                        and extra_sales * charged >= .90 * extra * cost
                        and option['profit'] >= best['profit'] - tolerance
                        and option['score'] >= best['score'] - tolerance
                        and option['loss'] <= best['loss'] + 2.):
                    supported.append(option)
            if supported:
                best = max(supported, key=lambda r: r['order']['meals'])
                best = dict(best, inventory_cushion=best['order']['meals']-base_meals,
                            base_meals=base_meals)
        return best

    def ceiling(self, rules, spot, quality, rep):
        cap = self.low_price_caps[quality] * self.spot_price_lift.get(spot, 1.) if self.small_crowds else rules.price_max
        return min(rules.price_max, self.price_cap or rules.price_max, cap,
                   reputation_value(quality, rep, rules.values))

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
        state = without_null_counts(state)
        self.observe(state)
        active_features = features(state.get('features'))
        self.learner.activate(active_features)
        if state.get('order_for_day') is None:
            next_day = (state.get('forecast') or {}).get('day') or int(state.get('day', 0))+1
            state = dict(state, order_for_day=next_day)
        rules = Rules.read(state)
        counts = self.forecast(state, rules)
        # Walking couples all campus crowds, but only eligible destinations can
        # become our order. Avoid expensive candidate searches that policy forbids.
        search_spots = rules.spots
        if self.small_crowds and active_features['walk_over']:
            search_spots = tuple(sorted(rules.spots, key=lambda s: (counts[s], s))[:3])
        digest = hashlib.sha256(repr((self.seed, self.market, state.get('order_for_day'))).encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], 'big'))
        self.scenario_rivals = []
        scenarios = self.scenarios(state, rules, counts, rng)
        results, coverage = [], {}
        by_pair = {}
        for spot in search_spots:
            rep = float((state.get('you') or {}).get('reputation', {}).get(spot, 50))
            coverage[spot] = []
            for quality in QUALITIES:
                coverage[spot].append(quality)
                ceiling = self.ceiling(rules, spot, quality, rep)
                low = max(rules.price_min, rules.costs[quality] + .25,
                          7.25 if self.small_crowds and quality == 'premium' else rules.price_min)
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
        for spot in search_spots:
            local = [r for (s, _), r in by_pair.items() if s == spot]
            if not local:
                continue
            top = max(local, key=lambda r: r['score'])
            quality, base = top['order']['quality'], top['order']['price']
            rep = float((state.get('you') or {}).get('reputation', {}).get(spot, 50))
            ceiling = self.ceiling(rules, spot, quality, rep)
            for step in (-.5, -.25, .25, .5):
                price = round(base + step, 2)
                floor = max(rules.price_min, rules.costs[quality]+.01,
                            7.25 if self.small_crowds and quality == 'premium' else rules.price_min)
                if floor <= price <= ceiling:
                    results.append(self.evaluate(state, rules, spot, quality, price, scenarios))
        if not results:
            raise ValueError('No legal candidates for supplied rules')
        restriction = self.location_restriction(results, counts, rules.spots)
        best = restriction['best']
        order = rules.validate(best['order'])
        self.pending = {'day': state.get('order_for_day'), 'forecast': state.get('forecast') or {},
                        'features': active_features,
                        'prediction': {'order': dict(order), 'sales': best['sales'],
                                       'features': active_features,
                                       'demand': best['demand'], 'profit': best['profit'],
                                       'rivals': sum(len(p[order['spot']]) for p, _, _, _ in scenarios)/len(scenarios)}}
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
                            'scenario_count': len(scenarios), 'prediction_checks': self.prediction_checks,
                            'features': active_features,
                            'rivals': {identity(t): self.opponents.summary(identity(t),
                                self.history.get(identity(t), []), state.get('forecast') or {},
                                self.history, rules.spots) for t in state.get('trucks') or []
                                if identity(t) != identity(state.get('you') or {})}}
        # Capture the same scenarios used for selection. This adds observability,
        # not another sample or a change to order scoring.
        evidence_model = dict(self.dump(), planner_config={
            'seed': self.seed, 'samples': self.samples, 'risk': self.risk,
            'small_crowds': self.small_crowds, 'price_cap': self.price_cap,
            'quiet_bonus': self.quiet_bonus, 'low_price_caps': dict(self.low_price_caps)})
        self.forecast_record = None
        try:
            self.forecast_record = capture(state, order, scenarios, self.scenario_rivals, evidence_model, self.diagnostics)
            self.pending['prediction']['forecast_id'] = self.forecast_record['forecast_id']
            self.pending['prediction']['rules_digest'] = self.forecast_record['rules_digest']
            self.pending['prediction']['regime_id'] = self.forecast_record['regime_id']
            self.diagnostics['risk'] = self.forecast_record['risk']
            self.diagnostics['forecast_id'] = self.forecast_record['forecast_id']
        except Exception as error:
            # Observability must never replace a valid order with a fallback.
            self.diagnostics['forecast_capture_error'] = type(error).__name__
        self.diagnostics['elapsed'] = time.perf_counter()-started
        return order
