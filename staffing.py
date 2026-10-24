"""Crew size and wage planning from the public labor market.

Mechanics (docs): each employee serves 50 meals and is paid the daily wage on days
we open. Each morning employees asking more than our wage quit, short rivals paying
at least max($3, 10%) more may lure employees (40% move), then we hire unemployed
workers who accept our wage. A new headcount or wage applies from the next hiring.

Learned online and persisted in the checkpoint:
- the unemployed pool's asking-wage curve, from labor.median/lowest_asking_wage;
- how many of our employees rivals actually poach per exposed rival-day;
- predicted versus realized crew fill, kept as checks.
Retention when lowering wages relies on a prior over employed asking wages, because
the pool of unemployed is selected for high asks and cannot describe our own crew.
"""
import math
from statistics import NormalDist

WAGE_MIN, WAGE_MAX, PER_EMPLOYEE = 5., 60., 50
POACH_PRIOR = (.5, 10.)  # events, exposed rival-days
EMPLOYED_ASK = NormalDist(5.6, 1.0)


def _binomial(n, p):
    return [math.comb(n, k) * p**k * (1-p)**(n-k) for k in range(n+1)]


class CrewModel:
    def __init__(self, data=None):
        data = data or {}
        self.pool = data.get('pool', {})
        self.poach = data.get('poach', {'events': POACH_PRIOR[0], 'exposures': POACH_PRIOR[1]})
        self.pending = data.get('pending')
        self.checks = data.get('checks', [])

    def to_dict(self):
        return {'pool': self.pool, 'poach': self.poach, 'pending': self.pending, 'checks': self.checks[-48:]}

    # --- learning -------------------------------------------------------
    def see_labor(self, state):
        """Fold in the pool stats once per day; planning also calls this, so a restart is not blind."""
        labor = state.get('labor')
        day = state.get('order_for_day') or state.get('day')
        if isinstance(labor, dict) and labor.get('median_asking_wage') is not None and self.pool.get('day') != day:
            old = self.pool.get('median')
            median = float(labor['median_asking_wage'])
            self.pool = {'median': median if old is None else .6*old + .4*median, 'day': day,
                         'lowest': float(labor.get('lowest_asking_wage', median)),
                         'unemployed': int(labor.get('unemployed', 0))}

    def observe(self, state, me):
        last = state.get('last_day') or {}
        self.see_labor(state)
        pending = self.pending
        if not pending or last.get('day') != pending['day']:
            return
        events = sum(int(e.get('count', 0)) for e in last.get('labor') or [] if e.get('from') == me)
        self.poach['events'] += events
        self.poach['exposures'] += pending['exposure']
        employees = (last.get('you') or {}).get('employees')
        if employees is not None:
            self.checks = (self.checks + [{
                'day': pending['day'], 'target': pending['target'], 'wage': pending['wage'],
                'predicted_employees': pending['expected_employees'], 'actual_employees': employees,
                'poached': events}])[-48:]
        self.pending = None

    # --- prediction -----------------------------------------------------
    def pool_accepting(self, wage, unemployed):
        """Unemployed workers expected to take this wage (normal curve through median/lowest)."""
        median = self.pool.get('median', 5.5)
        n = max(2, self.pool.get('unemployed', unemployed))
        spread = max(.3, (median - self.pool.get('lowest', median - 1.)) / NormalDist().inv_cdf(1 - .5/n))
        return unemployed * NormalDist(median, spread).cdf(wage)

    def exposure(self, state, wage, me):
        count = 0
        for truck in state.get('trucks') or []:
            other = truck.get('wage')
            if truck.get('id') != me and other is not None and wage <= float(other) - max(3., .1*float(other)):
                count += 1
        return count

    def slot_probability(self, state, headcount, wage, current_wage, me):
        high = EMPLOYED_ASK.cdf(max(current_wage, wage))
        low = EMPLOYED_ASK.cdf(WAGE_MIN - 1.)
        keep = 1. if wage >= current_wage else max(0., (EMPLOYED_ASK.cdf(wage) - low) / max(high - low, 1e-6))
        rate = self.poach['events'] / self.poach['exposures']
        poached = min(.9, rate * self.exposure(state, wage, me) / max(1, headcount))
        retained = keep * (1. - poached)
        labor = state.get('labor') or {}
        trucks = max(1, len(state.get('trucks') or []))
        refill = min(.97, self.pool_accepting(wage, int(labor.get('unemployed', 0))) / trucks)
        return retained + (1. - retained) * refill

    def search_space(self, state, current_wage, needed):
        """Headcounts and wages to consider, from the rules and what the market shows (no fixed ceiling).

        Wages run from the game's minimum to a dollar above the highest wage anyone pays (or the pool
        would ask), plus our own; headcounts run from what the order needs up to four spares. The
        expected-cost model decides inside this space, so spare staff and a higher wage are chosen only
        when the learned quit/poach risk pays for them."""
        rules = ((state.get('rules') or {}).get('staff')) or {}
        low = float(rules.get('wage_min', WAGE_MIN))
        cap = float(rules.get('wage_max', WAGE_MAX))
        seen = [float(t['wage']) for t in state.get('trucks') or [] if isinstance(t.get('wage'), (int, float))]
        spread = max(.3, (self.pool.get('median', 5.5) - self.pool.get('lowest', 4.5)) / 1.9) if self.pool else 1.
        pool_high = self.pool.get('median', 0.) + 3 * spread if self.pool else 0.
        high = min(cap, max([current_wage, low + 2.] + seen + [pool_high]) + 1.)
        grid = {round(low + .5 * i, 2) for i in range(int((high - low) / .5) + 1)} | {round(current_wage, 2)}
        grid |= {round(round(w * 2) / 2, 2) for w in seen if low <= w <= cap}
        staff_max = int(rules.get('staff_max') or 10)
        return range(needed, min(staff_max, needed + 4) + 1), sorted(w for w in grid if low <= w <= cap)

    def plan(self, state, order, sales, me):
        """Best (headcount, wage) for the accepted order, given scenario sales."""
        self.see_labor(state)
        crew = (state.get('you') or {}).get('staff') or {}
        current_h, current_w = int(crew.get('target', 4)), float(crew.get('wage', 6))
        meals = int(order['meals'])
        needed = max(1, math.ceil(meals / PER_EMPLOYEE))
        headcounts, wages = self.search_space(state, current_w, needed)
        margin = float(order['price']) * .98
        sales = sales or [meals * .8]

        def value(headcount, wage):
            p = self.slot_probability(state, headcount, wage, current_w, me)
            pmf = _binomial(headcount, p)
            wages = sum(q * k for k, q in enumerate(pmf)) * wage
            lost = sum(q * sum(max(0., min(s, meals) - PER_EMPLOYEE*k) for s in sales) / len(sales) * margin
                       for k, q in enumerate(pmf))
            return -wages - lost, p, pmf
        best = None
        for headcount in headcounts:
            for wage in wages:
                score, p, pmf = value(headcount, wage)
                if best is None or score > best[0]:
                    best = (score, headcount, wage, p, pmf)
        stay = value(current_h, current_w)[0] if current_h >= 1 else float('-inf')
        if best[0] < stay + 1.:  # hysteresis: only change for a clear gain
            score, headcount, wage = stay, current_h, current_w
            p, pmf = value(headcount, wage)[1:]
        else:
            score, headcount, wage, p, pmf = best
        enough = sum(q for k, q in enumerate(pmf) if PER_EMPLOYEE*k >= meals)
        expected = sum(q * k for k, q in enumerate(pmf))
        exposure = self.exposure(state, wage, me)
        self.pending = {'day': state.get('order_for_day'), 'target': headcount, 'wage': wage,
                        'exposure': exposure, 'expected_employees': expected}
        return {'headcount': headcount, 'wage': wage, 'change': (headcount, wage) != (current_h, current_w),
                'current': (current_h, current_w), 'needed': needed, 'prob_enough_staff': enough,
                'expected_employees': expected, 'expected_daily_wages': expected * wage,
                'poach_rate_per_exposed_rival': self.poach['events'] / self.poach['exposures'],
                'rivals_exposed': exposure, 'pool_median_ask': self.pool.get('median'),
                'unemployed': (state.get('labor') or {}).get('unemployed')}
