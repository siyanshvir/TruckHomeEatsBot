"""Adaptive policy for the final: learned calibration, dynamic location choice, bounded knobs.

The policy always observes (shadow learning on live days) but only steers decisions while
Brain.final_active. Every call is guarded: a failing component returns a neutral value and the
planner behaves like the baseline for that factor. Kill switch: NIGHTSHIFT_PROFILE=baseline or
data/profile.json {"profile": "baseline"} (see bot.profile_wanted).
"""
import json
from pathlib import Path
import time
from game import reputation_value, VALUES
from posture import maturity
from complaints import Complaints

try:
    from calibration import Calibrator
except ImportError:  # modules are optional so the baseline never depends on them
    Calibrator = None
try:
    from competition import Competition
except ImportError:
    Competition = None
try:
    from controller import MetaController, Knobs, load_priors
except ImportError:
    MetaController, Knobs, load_priors = None, None, None
try:
    from rules_adapter import RuleAdapter
except ImportError:
    RuleAdapter = None

MIDDLE = {'price_aggr': 1.0, 'breadth': .5, 'risk': .10}
SAMPLE_RANGE = (12, 32)
SLOW, FAST = .9, .4  # seconds of the last decision: shrink or grow the scenario count


class FinalPolicy:
    def __init__(self, data=None, priors=None, seed=0):
        data, priors = data or {}, priors or {}
        self.calibrator = self._build(Calibrator, data.get('calibration') or calibrator_seed(priors))
        self.competition = self._build(Competition, data.get('competition'))
        self.adapter = self._build(RuleAdapter, data.get('rules'))
        self.controller = None
        if MetaController is not None:
            try:
                self.controller = MetaController(data.get('controller'), seed=seed, priors=priors.get('controller'))
            except Exception:
                self.controller = None
        self.knobs = dict(MIDDLE)
        self.knob_day = None
        self.profit_mean = data.get('profit_mean')
        self.final_started = bool(data.get('final_started'))
        self.complaints = Complaints(data.get('complaints'))
        self.seed, self.priors = seed, priors
        self.errors = {}

    @staticmethod
    def _build(cls, data):
        if cls is None:
            return None
        try:
            return cls(data) if data else cls()
        except Exception:
            return None

    def _guard(self, name, fn, default):
        try:
            return fn()
        except Exception as error:
            self.errors[name] = type(error).__name__
            return default

    # --- persistence ----------------------------------------------------
    def to_dict(self):
        parts = {'calibration': self.calibrator, 'competition': self.competition,
                 'controller': self.controller, 'rules': self.adapter}
        out = {name: self._guard(name, lambda part=part: part.to_dict(), None)
               for name, part in parts.items() if part is not None}
        return dict(out, knobs=self.knobs, knob_day=self.knob_day, profit_mean=self.profit_mean, final_started=self.final_started,
                    complaints=self.complaints.to_dict())

    def restore(self, data):
        for key, cls, attr in (('calibration', Calibrator, 'calibrator'), ('competition', Competition, 'competition'),
                               ('rules', RuleAdapter, 'adapter')):
            if data.get(key) and cls is not None:
                setattr(self, attr, self._build(cls, data[key]) or getattr(self, attr))
        if data.get('controller') and MetaController is not None:
            self.controller = self._guard('controller', lambda: MetaController(data['controller']), self.controller)
        self.knobs = data.get('knobs') or dict(MIDDLE)
        self.knob_day = data.get('knob_day')
        self.profit_mean = data.get('profit_mean')
        self.final_started = bool(data.get('final_started'))
        self.complaints = Complaints(data.get('complaints'))

    def reset_for_final(self):
        """First final state: the tuner, competition and unexplained-profit residuals were learned under
        live conditions (reputation 100, one market), so they start fresh. Customer, crowd, rival,
        calibration, crew and event-rate learning lives elsewhere and is kept. Runs once, even across restarts."""
        if self.final_started:
            return False
        if MetaController is not None:
            self.controller = self._guard('reset', lambda: MetaController(None, seed=self.seed, priors=self.priors.get('controller')), self.controller)
        if Competition is not None:
            self.competition = self._build(Competition, None) or self.competition
        if RuleAdapter is not None:
            self.adapter = self._build(RuleAdapter, None) or self.adapter
        self.knobs, self.knob_day, self.profit_mean = dict(MIDDLE), None, None
        self.final_started = True
        return True

    def new_market(self):
        for part in (self.competition, self.controller):
            if part is not None:
                self._guard('new_market', lambda part=part: part.new_market(), None)

    # --- before the order -------------------------------------------------
    def begin(self, brain, state):
        """Pick tonight's knobs and apply risk and scenario count to the planner."""
        streak = 0
        for check in reversed([c for c in brain.prediction_checks
                               if c.get('mode') == state.get('mode') and c.get('market') == state.get('market')]):
            if (check.get('actual_profit') or 0) < 0:
                streak += 1
            else:
                break
        knobs = dict(MIDDLE)
        if self.controller is not None:
            context = dict(self._context(state), losing_streak=streak)  # the controller leans cautious on streaks
            chosen = self._guard('choose', lambda: self.controller.choose(context), None)
            if chosen is not None:
                knobs = {name: float(getattr(chosen, name)) for name in MIDDLE}
        self.knobs, self.knob_day = knobs, state.get('order_for_day')
        brain.risk = knobs['risk']
        elapsed = (brain.diagnostics or {}).get('elapsed')
        low, high = SAMPLE_RANGE
        if elapsed:
            if elapsed > SLOW:
                brain.samples = max(low, int(brain.samples * .7))
            elif elapsed < FAST and brain.samples < high:
                brain.samples = min(high, brain.samples + 4)

    @staticmethod
    def _context(state):
        return {'day': state.get('order_for_day'), 'days': state.get('days'), 'mode': state.get('mode')}

    def demand_factor(self, state, spot, quality, price, charged, rep):
        if self.calibrator is None:
            return 1.
        def run():
            values = (state.get('rules') or {}).get('values') or VALUES
            ceiling = reputation_value(quality, rep, values)
            factor = float(self.calibrator.factor(spot, quality, charged, ceiling))
            return max(.3, min(1.15, factor ** (1. / self.knobs['price_aggr'])))
        return self._guard('factor', run, 1.)

    def score_adjust(self, state, spot):
        """Dollars per day to subtract from a spot's score (negative is a bonus)."""
        total = 0.
        if self.competition is not None:
            def competition():
                penalty = float(self.competition.scaled(spot, state, self.knobs['breadth']))
                return penalty * maturity(state) if penalty < 0 else penalty   # roaming bonuses wait for a home spot
            total += self._guard('competition', competition, 0.)
        if self.adapter is not None:
            total += self._guard('spot_cost', lambda: float(self.adapter.spot_cost(state, spot, include_residual=False)), 0.)
        return max(-80., min(80., total))

    def order_extras(self, state, order):
        if self.adapter is None:
            return {}
        return self._guard('extras', lambda: dict(self.adapter.order_extras(state, order) or {}), {})

    def complaint_decision(self, state, brain, order):
        """Whether to file a health complaint tonight (see complaints.py); never raises."""
        recent = [c.get('actual_profit') for c in brain.prediction_checks[-7:] if c.get('actual_profit') is not None]
        daily = sum(recent) / len(recent) if recent else float((brain.diagnostics.get('chosen') or {}).get('profit') or 0.)
        return self._guard('complaints', lambda: self.complaints.analyse(state, brain, order, daily),
                           {'file': False, 'reason': 'error'})

    def is_wrecked(self, state):
        return bool(self.adapter is not None and self._guard('wrecked', lambda: self.adapter.is_wrecked(state), False))

    # --- after the day ----------------------------------------------------
    def observe_day(self, state, last, prediction, check, rep_before):
        """Learn from the completed day. Runs on live days too (shadow learning)."""
        order = check.get('order') or {}
        if not check.get('order_matches') or not order or check.get('surprise'):
            return
        spot, quality = order.get('spot'), order.get('quality')
        calibration = prediction.get('calibration') or 1.
        raw = (prediction.get('demand') or 0.) / max(calibration, 1e-6)
        values = (state.get('rules') or {}).get('values') or VALUES
        rep = float((rep_before or {}).get(spot, 50.))
        if self.calibrator is not None and check.get('actual_sales') is not None:
            self._guard('calibrate', lambda: self.calibrator.observe({
                'spot': spot, 'quality': quality, 'price': prediction.get('charged') or order.get('price'),
                'value_ceiling': reputation_value(quality, rep, values), 'expected_demand': raw,
                'actual_sales': check['actual_sales'], 'meals': order.get('meals'),
                'sellout': check.get('sellout'), 'weather': last.get('weather')}), None)
        if self.competition is not None:
            self._guard('compete', lambda: self.competition.observe(last, order, prediction), None)
        if self.adapter is not None:
            self._guard('rules', lambda: self.adapter.observe(last, order, prediction.get('profit')), None)
        if (self.controller is not None and self.knob_day == last.get('day')
                and check.get('actual_profit') is not None):
            knobs = Knobs(**self.knobs) if Knobs is not None else None
            actual = float(check['actual_profit']) + float(((last or {}).get('you') or {}).get('coupon_loss') or 0.)
            if knobs is not None:
                # Reward beats a running mean of real profit, which no knob can inflate by deflating the forecast.
                baseline = self.profit_mean if self.profit_mean is not None else actual
                self._guard('update', lambda: self.controller.update(knobs, actual - baseline), None)
            self.profit_mean = actual if self.profit_mean is None else .8 * self.profit_mean + .2 * actual

    def report(self):
        out = {'knobs': self.knobs, 'errors': dict(self.errors)}
        if self.calibrator is not None:
            out['calibration'] = self._guard('report', lambda: self.calibrator.report(), {})
        return out


def calibrator_seed(priors):
    """Calibrator state from train_final priors: a few pseudo-observations per price-ratio cell."""
    cells = {}
    try:
        from calibration import EDGES
        for quality, bins in ((priors.get('calibration') or {}).get('buckets') or {}).items():
            for span, cell in bins.items():
                low, high = (float(x) for x in span.split('-'))
                n = float(cell.get('n') or 0)
                if n <= 0:
                    continue
                key = '%s|%d' % (quality, sum((low + high) / 2 > edge for edge in EDGES))
                weight = min(n, 3.)
                total, old = cells.get(key, [0., 0.])
                cells[key] = [total + float(cell['mean']) * weight, old + weight]
    except Exception:
        return None
    return {'cells': cells, 'count': 0} if cells else None


def load_priors_file(path=None):
    path = Path(path) if path else Path(__file__).resolve().parent / 'model' / 'final_priors.json'
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def make_final_brain(model=None, samples=24, seed=0):
    """Brain with the adaptive policy attached and steering (used by final_eval and practice)."""
    from bot import make_brain
    brain = make_brain(model, samples=samples)
    attach_policy(brain, seed=seed)
    brain.use_final(True)
    return brain


def attach_policy(brain, seed=0, priors_path=None):
    """Attach a policy (inactive) so it can learn from live days before the final starts."""
    if brain.final_policy is None:
        brain.final_policy = FinalPolicy(priors=load_priors_file(priors_path), seed=seed)
    return brain.final_policy
