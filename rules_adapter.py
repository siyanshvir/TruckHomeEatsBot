"""Turns raw state/features into planner inputs so new rules never touch the planner.

Handles new spots, meat/veggie orders, riot exposure, coupon awareness and learned
unmodeled profit terms. Never raises on malformed input; missing data is neutral.
"""
import copy
import math

from game import DISTANCES, PRIORS, SPOTS

# New-spot hypotheses: (price limit, premium share, crowd weight, meat-only, veggie-only).
NEW_SPOTS = {
    'fisher': (14., .30, .12, .40, .10),
    'kottman': (12.5, .25, .08, .10, .50),
    'mirror': (12., .20, .10, .25, .20),
    'medcenter': (13., .20, .14, .25, .15),
    'lane': (12., .15, .12, .30, .10),
    'rpac': (12., .25, .09, .45, .10),
    'schott': (13., .20, .07, .35, .10),
    'gateway': (11.5, .15, .11, .25, .15),
}
BASE_DIETS = {'oval': (.20, .12), 'union': (.20, .12), 'northres': (.20, .12), 'thompson': (.15, .20),
              'stadium': (.27, .03), 'highst': (.24, .06)}
RIOTS = {'loss': (.7, 260., 'highst'), 'title': (.95, 380., 'stadium')}
GENERIC_PRIOR = {'mu': 12., 'sd': 3., 'mix': (.45, .40, .15), 'weight': .10}


def _num(x, default=None):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _dict(x):
    return x if isinstance(x, dict) else {}


def _list(x):
    return x if isinstance(x, (list, tuple)) else []


def _open_spots(state):
    spots = _dict(state.get('rules')).get('spots')
    return [s for s in _list(spots) if isinstance(s, str)] or list(SPOTS)


def prior_for(spot):
    if spot in PRIORS:
        mu, sd, mix, weight = PRIORS[spot]
        return {'mu': mu, 'sd': sd, 'mix': tuple(mix), 'weight': weight}
    if spot in NEW_SPOTS:
        limit, premium, weight = NEW_SPOTS[spot][:3]
        cheap = max(.05, min(.6, .55 - premium))
        return {'mu': limit, 'sd': 3., 'mix': (cheap, round(1. - cheap - premium, 3), premium), 'weight': weight}
    return dict(GENERIC_PRIOR)


def _coords(state):
    out = {}
    for spot, info in _dict(state.get('spot_info')).items():
        info = _dict(info)
        x, z = _num(info.get('x')), _num(info.get('z'))
        if x is not None and z is not None:
            out[spot] = (x, z)
    return out


def distances_for(state):
    spots = _open_spots(state)
    coords = _coords(state)
    out = {}
    for a in spots:
        for b in spots:
            if a == b:
                continue
            if a in coords and b in coords:
                out[(a, b)] = math.hypot(coords[a][0] - coords[b][0], coords[a][1] - coords[b][1])
            elif b in DISTANCES.get(a, {}):
                out[(a, b)] = float(DISTANCES[a][b])
    return out


def _riot_law_scale(state):
    """Election laws scale riot fierceness (crackdown 0, tailgate 1.5); 1 when no law says so."""
    try:
        from riots import law_multiplier
        return max(0., float(law_multiplier(state)))
    except Exception:
        return 1.


class RuleAdapter:
    def __init__(self, data=None):
        data = _dict(data)
        self.residual = {k: float(v) for k, v in _dict(data.get('residual')).items() if _num(v) is not None}
        self.streak = {k: int(v) for k, v in _dict(data.get('streak')).items() if _num(v) is not None}
        self.profit_proxy = _num(data.get('profit_proxy'), 150.)
        self.coupons_seen = int(_num(data.get('coupons_seen'), 0))
        self.wrecks = int(_num(data.get('wrecks'), 0))
        self.hits = int(_num(data.get('hits'), 0))
        self.min_streak = 3
        self.trivial = 3.

    def to_dict(self):
        return {'residual': dict(self.residual), 'streak': dict(self.streak),
                'profit_proxy': self.profit_proxy, 'coupons_seen': self.coupons_seen,
                'wrecks': self.wrecks, 'hits': self.hits}

    # ---- state normalization -------------------------------------------------
    def normalize(self, state):
        try:
            out = copy.deepcopy(state) if isinstance(state, dict) else {}
            out['spot_priors'] = {s: prior_for(s) for s in _open_spots(out)}
            out['spot_distances'] = distances_for(out)
            fc = out.get('forecast')
            if isinstance(fc, dict) and isinstance(fc.get('expected_customers'), dict):
                fc['expected_customers'] = {k: v for k, v in fc['expected_customers'].items()
                                            if _num(v) is not None}
            return out
        except Exception:
            return dict(state) if isinstance(state, dict) else {}

    # ---- meat or veggie ------------------------------------------------------
    def order_extras(self, state, order):
        try:
            if not _dict(_dict(state).get('features')).get('food'):
                return {}
            spot = _dict(order).get('spot')
            diets = _dict(_dict(_dict(state).get('forecast')).get('diets')).get(spot)
            if isinstance(diets, dict):
                meat, veg = _num(diets.get('meat')), _num(diets.get('veggie'))
            else:
                meat = veg = None
            if meat is None or veg is None:
                base = BASE_DIETS.get(spot) or (NEW_SPOTS[spot][3:5] if spot in NEW_SPOTS else None)
                if not base:
                    return {'food': 'meat'}
                meat, veg = base
            rivals = {'meat': 0, 'veggie': 0}
            for t in _list(_dict(_dict(_dict(state).get('last_day')).get('spots')).get(spot, {}).get('trucks')
                           if isinstance(_dict(_dict(state).get('last_day')).get('spots'), dict) else []):
                f = _dict(t).get('food')
                if f in rivals:
                    rivals[f] += 1
                elif isinstance(t, dict):
                    rivals['meat'] += 1
            # share per truck already serving that diet (+1 for us); higher means underserved
            meat_score = meat / (rivals['meat'] + 1.)
            veg_score = veg / (rivals['veggie'] + 1.)
            return {'food': 'veggie' if veg_score > meat_score else 'meat'}
        except Exception:
            return {}

    # ---- riots ---------------------------------------------------------------
    def _riot_warnings(self, state):
        fc = _dict(_dict(state).get('forecast'))
        entries = []
        for key in ('riot', 'riots'):
            raw = fc.get(key)
            if isinstance(raw, dict):
                entries.append(raw)
            elif isinstance(raw, (list, tuple)):
                entries.extend(raw)
            elif isinstance(raw, str):
                entries.append(raw)
            elif raw:
                entries.append('riot')
        for e in _list(fc.get('events')):
            if 'riot' in str(e).lower():
                entries.append(e)
        out = []
        for e in entries:
            text = str(e).lower()
            kind = 'title' if any(w in text for w in ('title', 'national', 'champion')) else 'loss'
            intensity, radius, center = RIOTS[kind]
            if isinstance(e, dict):
                intensity = _num(e.get('intensity'), intensity)
                radius = _num(e.get('radius'), radius)
                spot = e.get('spot') or e.get('where') or e.get('location')
                if isinstance(spot, str) and spot:
                    center = spot
            out.append((intensity, radius or 1., center, kind))
        return out

    def _repair_cost(self, d, radius):
        return 30. + 120. * math.exp(-d / radius)

    def spot_cost(self, state, spot, include_residual=True):
        try:
            cost = 0.
            if _dict(_dict(state).get('features')).get('riots'):
                scale = _riot_law_scale(state)
                dist = distances_for(state if isinstance(state, dict) else {})
                for intensity, radius, center, _ in self._riot_warnings(state):
                    d = 0. if spot == center else dist.get((spot, center), dist.get((center, spot)))
                    if d is None:
                        continue
                    p = min(1., max(0., intensity * math.exp(-d / radius)))
                    cost += scale * p * (.3 * self.profit_proxy + .7 * self._repair_cost(d, radius))
            if include_residual and self.streak.get(spot, 0) >= self.min_streak:
                cost -= self.residual.get(spot, 0.)
            return cost
        except Exception:
            return 0.

    # ---- learning ------------------------------------------------------------
    def observe(self, last_day, our_order, modeled_profit):
        try:
            you = _dict(_dict(last_day).get('you'))
            spot = _dict(our_order).get('spot')
            if you.get('coupons') or _num(you.get('coupon_share'), 0.) > 0 or _num(you.get('coupon_loss'), 0.) > 0:
                self.coupons_seen += 1
            if you.get('wrecked_until_day') or you.get('wrecked'):
                self.wrecks += 1
            repairs = _num(you.get('repairs'), 0.) or 0.
            if repairs > 0:
                self.hits += 1
            reported, modeled = _num(you.get('profit')), _num(modeled_profit)
            if reported is None:
                return
            if reported > 0:
                self.profit_proxy = .8 * self.profit_proxy + .2 * reported
            if modeled is None or not isinstance(spot, str):
                return
            # repairs and coupon loss are explained terms; the rest is unmodeled
            res = reported - modeled + repairs + (_num(you.get('coupon_loss'), 0.) or 0.)
            if abs(res) <= self.trivial:
                self.streak[spot] = 0
                self.residual[spot] = .5 * self.residual.get(spot, 0.)
                return
            prev = self.residual.get(spot)
            same = prev is not None and prev * res > 0
            self.streak[spot] = self.streak.get(spot, 0) + 1 if same or prev is None else 1
            self.residual[spot] = res if prev is None or not same else .6 * prev + .4 * res
        except Exception:
            return

    # ---- coupons / wrecks ----------------------------------------------------
    def coupon_action(self, state):
        """Hook for offensive coupons. Defensive stance: always None until a measured gain exists."""
        return None

    def is_wrecked(self, state):
        try:
            state = _dict(state)
            until = _num(_dict(state.get('you')).get('wrecked_until_day'))
            day = _num(state.get('order_for_day'))
            return until is not None and day is not None and until >= day
        except Exception:
            return False
