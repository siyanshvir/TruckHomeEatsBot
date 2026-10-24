"""Independent, deliberately imperfect stress simulator; not the real game engine.

Random environments are pre-generated so candidate policies see matched customers.
Use server practice to validate; offline profit is only a regression/stress signal.
"""
import math
import random
from game import SPOTS, QUALITIES, COSTS, VALUES, REP_CHANGE, Rules

# Separate truth parameters rather than the learner's fitted distributions.
TRUTH = {'highst': (12, 4, (.52, .33, .15), .24), 'oval': (13, 3, (.38, .42, .20), .20),
         'union': (13, 3, (.18, .62, .20), .18), 'northres': (11, 2.5, (.48, .44, .08), .16),
         'thompson': (12, 3, (.45, .40, .15), .14), 'stadium': (13, 3, (.25, .50, .25), .08)}


class Market:
    def __init__(self, seed, rivals=5, days=30, style='mixed'):
        self.seed, self.days, self.style = seed, days, style
        self.names = ['us'] + ['r%d' % i for i in range(rivals)]
        self.reps = {name: dict.fromkeys(SPOTS, 50.) for name in self.names}
        self.cash = dict.fromkeys(self.names, 0.)
        self.day, self.last = 1, None
        rng = random.Random(str(seed))
        self.truth = {s: (v[0]+rng.uniform(-2, 2), v[1]*rng.uniform(.8, 1.3), v[2], v[3]) for s, v in TRUTH.items()}
        self.schedule = []
        for day in range(days):
            odds = {'clear': .4, 'cloudy': .3, 'rain': .2, 'cold': .1}
            weather = rng.choices(list(odds), weights=list(odds.values()))[0]
            event = rng.choice(['none', 'none', 'game', 'midterm', 'concert', 'late'])
            spot_customers, expected = {}, {}
            for spot, (mu, sd, mix, share) in self.truth.items():
                multiplier = 1.
                if event == 'game': multiplier = 5. if spot == 'stadium' else 1.2
                if event == 'midterm' and spot == 'thompson': multiplier = 1.7; mu += .5; mix = (.1, .35, .55)
                if event == 'concert' and spot == 'union': multiplier = 1.8; mu += 1.
                if event == 'late':
                    multiplier = 2. if spot in ('highst', 'northres') else .5
                    if spot in ('highst', 'northres'): mix = (.75, .20, .05)
                factors = {'clear': 1., 'cloudy': .95, 'rain': .55 if spot in ('oval', 'stadium') else .85,
                           'cold': .8 if spot in ('oval', 'stadium') else .95}
                base = 110 * max(4, len(self.names)) * share * multiplier
                expected[spot] = round(base * sum(p*factors[w] for w, p in odds.items()))
                actual = int(max(0, rng.gauss(base*factors[weather], .08*base)))
                customers = []
                for _ in range(actual):
                    # Mixture tails intentionally violate the policy's normal-budget assumption.
                    budget = rng.gauss(mu, sd) + (rng.uniform(-3, 3) if rng.random() < .15 else 0)
                    taste = rng.random()
                    minimum = 0 if taste < mix[0] else 1 if taste < mix[0]+mix[1] else 2
                    customers.append((budget, minimum))
                spot_customers[spot] = customers
            self.schedule.append(({'weather': odds, 'events': [] if event == 'none' else [event],
                                   'expected_customers': expected}, weather, spot_customers))

    def state(self):
        return {'version': self.day, 'mode': 'practice', 'market': str(self.seed), 'days': self.days,
                'day': self.day-1, 'order_for_day': self.day, 'phase': 'night' if self.day <= self.days else 'over',
                'orders_open': self.day <= self.days,
                'forecast': self.schedule[min(self.day-1, self.days-1)][0],
                'rules': {'budget': 500, 'costs': COSTS, 'values': VALUES, 'spots': list(SPOTS), 'price_min': 1, 'price_max': 30},
                'you': {'id': 'us', 'cash': self.cash['us'], 'reputation': dict(self.reps['us'])},
                'last_day': self.last,
                'trucks': [{'id': name, 'reputation_avg': sum(self.reps[name].values())/6, 'cash': self.cash[name]} for name in self.names]}

    def rival_orders(self):
        forecast = self.schedule[self.day-1][0]['expected_customers']
        orders = {}
        for index, name in enumerate(self.names[1:]):
            rng = random.Random('%s/%s/%s' % (self.seed, self.day, name))
            if self.style == 'crowd':
                spot = max(forecast, key=forecast.get)
            elif self.style == 'moving':
                spot = rng.choices(SPOTS, weights=[forecast[s]+20 for s in SPOTS])[0]
            else:
                spot = SPOTS[index % 6] if rng.random() < .7 else rng.choice(SPOTS)
            if self.style == 'brutal':
                orders[name] = self.brutal_order(index, name, forecast, rng)
                continue
            quality = ('premium', 'standard', 'cheap')[index % 3]
            price = {'premium': 9.25, 'standard': 7.25, 'cheap': 5.25}[quality]
            if self.style == 'moving': price += rng.uniform(-1, 1)
            if self.style in ('undercut', 'cycle'):
                # Independently implemented stress opponents inspired by Price Wars'
                # documented Cheapest / Two Bound examples, using only past public data.
                previous = ((self.last or {}).get('spots', {}).get(spot) or {}).get('trucks', [])
                comparable = [t['price'] for t in previous if t['id'] != name and t['quality'] == quality]
                if comparable:
                    floor = COSTS[quality] + .5
                    price = max(floor, min(comparable)-.10)
                    if self.style == 'cycle' and price <= floor:
                        price = {'cheap': 7.5, 'standard': 11., 'premium': 14.}[quality]
            orders[name] = {'spot': spot, 'quality': quality, 'price': round(price, 2),
                            'meals': int(500/COSTS[quality]) if index % 2 == 0 else 55}
        return orders

    def brutal_order(self, index, name, forecast, rng):
        """Aggressive premium rivals: campers, a hunter that shadows and undercuts us, a copycat of
        yesterday's winner, a price-war truck, a crowd chaser and a mover. All use only public data."""
        kind = index % 6
        crowd = max(forecast, key=forecast.get)
        mine = (self.last or {}).get('you') or {}
        if kind == 0:     # camper: one home spot, premium just under $10
            return {'spot': SPOTS[(index // 6 + index) % 6], 'quality': 'premium', 'price': round(9.4 + rng.random()*.5, 2), 'meals': 90}
        if kind == 1:     # hunter: parks where we parked and undercuts us by a nickel
            if mine.get('spot'):
                return {'spot': mine['spot'], 'quality': 'premium', 'price': round(max(8., min(11., mine['price']) - .05), 2), 'meals': 90}
            return {'spot': crowd, 'quality': 'premium', 'price': 9.5, 'meals': 90}
        if kind == 2:     # copycat: repeats yesterday's most profitable order
            best = None
            for spot, info in ((self.last or {}).get('spots') or {}).items():
                for t in info['trucks']:
                    gain = t['sold']*t['price'] - t['meals']*COSTS[t['quality']]
                    if best is None or gain > best[0]:
                        best = (gain, spot, t)
            if best:
                t = best[2]
                return {'spot': best[1], 'quality': t['quality'], 'price': t['price'], 'meals': min(t['meals'], int(500/COSTS[t['quality']]))}
            return {'spot': crowd, 'quality': 'premium', 'price': 9.75, 'meals': 90}
        if kind == 3:     # price war: premium near cost wherever the crowd is
            return {'spot': crowd, 'quality': 'premium', 'price': 8.5, 'meals': 90}
        if kind == 4:     # chaser: biggest crowd, mid price
            return {'spot': crowd, 'quality': 'premium', 'price': 9.75, 'meals': 90}
        spots = rng.choices(SPOTS, weights=[forecast[s] + 20 for s in SPOTS])[0]
        return {'spot': spots, 'quality': 'premium', 'price': round(rng.uniform(9, 10.5), 2), 'meals': 90}

    def play(self, own_order):
        rules = Rules.read(self.state())
        rules.validate(own_order)
        orders = self.rival_orders()
        orders['us'] = own_order
        forecast, weather, customers = self.schedule[self.day-1]
        effective = max(4., sum(min(1., o['meals']*COSTS[o['quality']]/250) for o in orders.values()))
        scale = effective/max(4, len(self.names))
        spots, own = {}, None
        for spot in SPOTS:
            here = {name: o for name, o in orders.items() if o['spot'] == spot}
            sold = dict.fromkeys(here, 0)
            crowd = customers[spot][:round(len(customers[spot])*scale)]
            for budget, minimum in crowd:
                eligible = []
                for name, o in here.items():
                    q = QUALITIES.index(o['quality'])
                    value = VALUES[o['quality']]*(.7+.006*self.reps[name][spot])-o['price']
                    if q >= minimum and o['price'] <= budget and sold[name] < o['meals'] and value >= 0:
                        eligible.append((value, -o['price'], name))
                if eligible:
                    sold[max(eligible)[2]] += 1
            records = []
            for name, o in here.items():
                profit = sold[name]*o['price']-o['meals']*COSTS[o['quality']]
                self.cash[name] += profit
                records.append(dict(o, id=name, sold=sold[name]))
                if name == 'us': own = dict(o, sold=sold[name], profit=profit, spoiled=o['meals']-sold[name])
            spots[spot] = {'customers': len(crowd), 'hungry': len(crowd)-sum(sold.values()), 'trucks': records}
        for name, o in orders.items():
            for spot in SPOTS:
                if spot == o['spot']:
                    record = next(t for t in spots[spot]['trucks'] if t['id'] == name)
                    if record['sold']:
                        self.reps[name][spot] = max(0, min(100, self.reps[name][spot]+REP_CHANGE[o['quality']]))
                else:
                    self.reps[name][spot] += .03*(50-self.reps[name][spot])
        self.last = {'day': self.day, 'weather': weather, 'events': forecast['events'], 'spots': spots, 'you': own}
        self.day += 1
        return self.state()
