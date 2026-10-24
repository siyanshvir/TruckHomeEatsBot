"""Learned per-spot adjustment replacing a hard "smallest crowds" location cap.

penalty(spot, state) returns dollars/day to subtract from a spot's score (negative = bonus):
- shortfall: EMA of (our realized profit - predicted profit) at the spot, shrunk to 0 by a
  pseudo-count; usually underperforming the plan there is a cost;
- pressure: recent rival count, how often trucks showed up at a spot right after we left it
  (follow behaviour) and how deep they undercut our price;
- exploration bonus: spots not visited in this market while reputation is still ~50.
Everything is bounded and 0 with no data (except the small bonus for unvisited spots).
"""
import math

LIMIT = 60.
SHRINK = 2.            # pseudo-count for the shortfall EMA
ALPHA = .3             # EMA rate
SHORT_MAX, SHORT_MIN = 40., -15.
PRESSURE_MAX = 25.
EXPLORE_MAX = 12.
REP_BAND = 1.5
TOP_RIVALS = 1.


def _num(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _dict(value):
    return value if isinstance(value, dict) else {}


def _ema(old, new, count):
    rate = max(ALPHA, 1. / (count + 1))
    return new if old is None else old + rate * (new - old)


class Competition:
    def __init__(self, data=None):
        data = _dict(data)
        self.spots = {}
        for spot, row in _dict(data.get('spots')).items():
            row = _dict(row)
            self.spots[str(spot)] = {
                'gap': _num(row.get('gap'), 0.), 'n': max(0, int(_num(row.get('n'), 0))),
                'rivals': _num(row.get('rivals'), 0.), 'rn': max(0, int(_num(row.get('rn'), 0))),
                'follow': _num(row.get('follow'), 0.), 'fn': max(0, int(_num(row.get('fn'), 0))),
                'undercut': _num(row.get('undercut'), 0.),
                'visits': max(0, int(_num(row.get('visits'), 0)))}
        self.prev_loc = {str(k): str(v) for k, v in _dict(data.get('prev_loc')).items()}
        self.last_spot = data.get('last_spot') if isinstance(data.get('last_spot'), str) else None
        self.last_price = _num(data.get('last_price'))

    def to_dict(self):
        return {'spots': {spot: dict(row) for spot, row in self.spots.items()},
                'prev_loc': dict(self.prev_loc), 'last_spot': self.last_spot,
                'last_price': self.last_price}

    def row(self, spot):
        return self.spots.setdefault(spot, {'gap': 0., 'n': 0, 'rivals': 0., 'rn': 0, 'follow': 0.,
                                            'fn': 0, 'undercut': 0., 'visits': 0})

    def new_market(self):
        """Visits and positional memory are per market; learned gaps and pressure carry (halved weight)."""
        for row in self.spots.values():
            row['visits'] = 0
            row['n'] = row['n'] // 2
            row['rn'] = row['rn'] // 2
            row['fn'] = row['fn'] // 2
        self.prev_loc = {}
        self.last_spot = None
        self.last_price = None

    # --- learning ---------------------------------------------------------
    @staticmethod
    def _trucks(day, spot):
        trucks = _dict(_dict(_dict(day).get('spots')).get(spot)).get('trucks')
        return [t for t in trucks if isinstance(t, dict)] if isinstance(trucks, list) else []

    def observe(self, last_day, our_order, predicted):
        try:
            self._observe(_dict(last_day), _dict(our_order), _dict(predicted))
        except (TypeError, ValueError, KeyError, AttributeError):
            pass

    def _observe(self, last_day, order, predicted):
        spot = order.get('spot')
        spots = _dict(last_day.get('spots'))
        if not isinstance(spot, str) or spot not in spots:
            return
        you = _dict(last_day.get('you'))
        my_id = you.get('id', order.get('id'))
        price = _num(order.get('price'))
        row = self.row(spot)
        row['visits'] += 1

        def ours(truck):
            if my_id is not None and truck.get('id') == my_id:
                return True
            return my_id is None and price is not None and _num(truck.get('price')) == price and \
                _num(truck.get('meals')) == _num(order.get('meals'))

        # (a) shortfall versus the planner's own number
        realized = _num(you.get('profit'), _num(last_day.get('profit')))
        expected = _num(predicted.get('profit'))
        if realized is not None and expected is not None:
            row['gap'] = _ema(row['gap'] if row['n'] else None, max(-300., min(300., realized - expected)), row['n'])
            row['n'] += 1

        # (b) rival pressure where we sold, and follow behaviour where we left
        rivals_here = [t for t in self._trucks(last_day, spot) if not ours(t)]
        row['rivals'] = _ema(row['rivals'] if row['rn'] else None, float(len(rivals_here)), row['rn'])
        row['rn'] += 1
        if price is not None:
            gaps = [price - p for p in (_num(t.get('price')) for t in rivals_here) if p is not None]
            if gaps:
                row['undercut'] = _ema(row['undercut'], max(0., max(gaps)), row['n'])
        left = self.last_spot
        if left and left != spot and left in spots:
            arrivals = [t for t in self._trucks(last_day, left)
                        if not ours(t) and t.get('id') is not None and self.prev_loc.get(str(t['id'])) not in (None, left)]
            lrow = self.row(left)
            lrow['follow'] = _ema(lrow['follow'] if lrow['fn'] else None, min(3., float(len(arrivals))), lrow['fn'])
            lrow['fn'] += 1
            last_price = self.last_price
            if last_price is not None:
                gaps = [last_price - p for p in (_num(t.get('price')) for t in arrivals) if p is not None]
                if gaps:
                    lrow['undercut'] = _ema(lrow['undercut'], max(0., max(gaps)), lrow['fn'])
        loc = {}
        for name in spots:
            for truck in self._trucks(last_day, name):
                if truck.get('id') is not None and not ours(truck):
                    loc[str(truck['id'])] = name
        self.prev_loc = loc
        self.last_spot = spot
        self.last_price = price

    # --- scoring ----------------------------------------------------------
    def penalty(self, spot, state=None):
        try:
            return self._penalty(spot, _dict(state))
        except (TypeError, ValueError, KeyError, AttributeError):
            return 0.

    def _penalty(self, spot, state):
        row = self.spots.get(spot)
        total = 0.
        if row:
            shortfall = -row['gap'] * row['n'] / (row['n'] + SHRINK)
            total += max(SHORT_MIN, min(SHORT_MAX, shortfall))
            rival_conf = row['rn'] / (row['rn'] + SHRINK)
            follow_conf = row['fn'] / (row['fn'] + SHRINK)
            pressure = 2. * max(0., row['rivals'] - TOP_RIVALS) * rival_conf \
                + 10. * row['follow'] * follow_conf \
                + 3. * min(3., row['undercut']) * max(rival_conf, follow_conf)
            total += min(PRESSURE_MAX, pressure)
        visits = row['visits'] if row else 0
        rep = _num(_dict(_dict(state.get('you')).get('reputation')).get(spot), 50.)
        if abs(rep - 50.) <= REP_BAND:
            day = _num(state.get('order_for_day'), _num(state.get('day'), 1.))
            days = max(1., _num(state.get('days'), 30.))
            remaining = max(0., min(1., (days - day + 1.) / days))
            total -= EXPLORE_MAX * remaining / (1. + visits) ** 2
        return max(-LIMIT, min(LIMIT, total))

    def scaled(self, spot, state, breadth):
        b = _num(breadth, 0.)
        return max(-LIMIT, min(LIMIT, b * self.penalty(spot, state)))
