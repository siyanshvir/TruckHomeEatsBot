"""Meta-controller: a bounded tuner of three coarse knobs.

Each knob is an independent discounted Thompson-sampling bandit over three arms with Gaussian
posteriors. The reward is the realized daily profit minus the model's own predicted profit,
divided by a running scale so it is roughly unit size: the bandits learn which settings the
planner over- or under-trusts. All three knobs are credited with the same reward.

Per arm we keep discounted sufficient statistics (n, s) on top of a prior (mean, var):
    precision = 1/prior_var + n/OBS_VAR,  mean = (prior_mean/prior_var + s/OBS_VAR)/precision
Every update decays the evidence of every arm, so unplayed arms grow uncertain again and the
controller can track regime shifts. It never leaves its arm sets.
"""
import json
import math
import random
from typing import NamedTuple


class Knobs(NamedTuple):
    price_aggr: float
    breadth: float
    risk: float


ARMS = {'price_aggr': (0.90, 1.00, 1.10), 'breadth': (0.0, 0.5, 1.0), 'risk': (0.03, 0.10, 0.30)}
NAMES = tuple(ARMS)
DEFAULT = Knobs(1.00, 0.5, 0.10)
GAMMA = 0.97          # evidence discount per update
FORGET = 0.5          # evidence kept at a new market
OBS_VAR = 1.0
MID_PRIOR = (0.6, 0.16)    # (mean, var) of the middle arm: favoured, so it wins most draws
SIDE_PRIOR = (0.0, 0.16)
VAR_FLOOR, VAR_CEIL = 0.01, 4.0
SCALE0, SCALE_MIN = 30., 1.
CLIP = 3.


def _num(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _key(arm):
    return repr(float(arm))


def load_priors(path):
    """Read model/final_priors.json's 'controller' section; {} when missing or malformed."""
    try:
        with open(path) as handle:
            data = json.load(handle)
        section = data.get('controller')
        return section if isinstance(section, dict) else {}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def _prior(name, index, priors):
    mean, var = MID_PRIOR if index == 1 else SIDE_PRIOR
    arm = ARMS[name][index]
    entry = {}
    try:
        group = (priors or {}).get(name) or {}
        entry = group.get(str(arm)) or group.get(_key(arm)) or group.get('%g' % arm) or {}
    except AttributeError:
        pass
    if isinstance(entry, dict):
        mean = _num(entry.get('mean'), mean)
        var = _num(entry.get('var'), var)
    return mean, min(max(var, VAR_FLOOR), VAR_CEIL)


def _fresh(priors=None):
    out = {}
    for name in NAMES:
        out[name] = []
        for index in range(3):
            mean, var = _prior(name, index, priors)
            out[name].append({'pm': mean, 'pv': var, 'n': 0., 's': 0.})
    return out


class MetaController:
    def __init__(self, data=None, seed=0, priors=None):
        data = data if isinstance(data, dict) else {}
        self.arms = _fresh(priors)
        stored = data.get('arms')
        if isinstance(stored, dict):
            for name in NAMES:
                rows = stored.get(name)
                if not isinstance(rows, list) or len(rows) != 3:
                    continue
                for index, row in enumerate(rows):
                    if not isinstance(row, dict):
                        continue
                    arm = self.arms[name][index]
                    for field in ('pm', 'pv', 'n', 's'):
                        value = _num(row.get(field))
                        if value is not None:
                            arm[field] = value
                    arm['pv'] = min(max(arm['pv'], VAR_FLOOR), VAR_CEIL)
                    arm['n'] = max(0., arm['n'])
        self.scale_sq = max(_num(data.get('scale_sq'), SCALE0 ** 2), SCALE_MIN ** 2)
        self.updates = max(0, int(_num(data.get('updates'), 0)))
        self.seed = int(_num(data.get('seed'), seed))
        self.rng = random.Random(self.seed)
        try:
            version, internal, gauss = data['rng']
            self.rng.setstate((int(version), tuple(int(x) for x in internal), gauss))
        except (KeyError, TypeError, ValueError):
            pass

    def to_dict(self):
        return {'arms': {name: [dict(row) for row in rows] for name, rows in self.arms.items()},
                'scale_sq': self.scale_sq, 'updates': self.updates, 'seed': self.seed,
                'rng': list(self.rng.getstate())}

    # --- posterior --------------------------------------------------------
    @staticmethod
    def posterior(row):
        precision = 1. / row['pv'] + row['n'] / OBS_VAR
        return (row['pm'] / row['pv'] + row['s'] / OBS_VAR) / precision, 1. / precision

    def means(self):
        return {name: [self.posterior(row)[0] for row in rows] for name, rows in self.arms.items()}

    # --- choosing ---------------------------------------------------------
    def choose(self, context=None, greedy=False):
        streak = 0
        if isinstance(context, dict):
            streak = min(3, max(0, int(_num(context.get('losing_streak'), 0))))
        picks = []
        for name in NAMES:
            scores = []
            for index, row in enumerate(self.arms[name]):
                mean, var = self.posterior(row)
                if not greedy:
                    mean = self.rng.gauss(mean, math.sqrt(var))
                if name == 'risk' and streak:   # recent losing streak: temporarily lean cautious
                    mean += 0.15 * streak / 3. * (index - 1)
                scores.append(mean)
            best = max(range(3), key=lambda i: (scores[i], i == 1))
            picks.append(ARMS[name][best])
        return Knobs(*picks)

    # --- learning ---------------------------------------------------------
    def normalize(self, reward):
        reward = _num(reward)
        if reward is None:
            return None
        scale = math.sqrt(self.scale_sq)
        out = max(-CLIP, min(CLIP, reward / scale))
        rate = max(0.05, 1. / (self.updates + 2))
        self.scale_sq = max(SCALE_MIN ** 2, (1 - rate) * self.scale_sq + rate * reward * reward)
        return out

    def update(self, knobs, reward):
        """Credit all three knobs with the (normalized) reward; returns the normalized value."""
        try:
            values = tuple(knobs)
            if len(values) != 3:
                return None
        except TypeError:
            return None
        norm = self.normalize(reward)
        if norm is None:
            return None
        for name, value in zip(NAMES, values):
            value = _num(value)
            if value is None:
                continue
            index = min(range(3), key=lambda i: abs(ARMS[name][i] - value))
            for row in self.arms[name]:
                row['n'] *= GAMMA
                row['s'] *= GAMMA
            self.arms[name][index]['n'] += 1.
            self.arms[name][index]['s'] += norm
        self.updates += 1
        return norm

    def new_market(self):
        """Partial forgetting: shrink evidence toward the prior; the learned means are kept."""
        for rows in self.arms.values():
            for row in rows:
                row['n'] *= FORGET
                row['s'] *= FORGET
