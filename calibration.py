"""Online correction of the planner's uncapped demand by price level.

Each observation gives target = actual_sales / expected_demand in a cell
(quality, price/value_ceiling bin). A cell shrinks toward the pooled estimate
of cells at the same or a lower price ratio (same quality, then all qualities),
so misses at high prices never drag down low-price factors.
"""
import math

EDGES = (.55, .7, .85)       # ratio bins: <=.55, .55-.7, .7-.85, >.85
QUALITIES = ('cheap', 'standard', 'premium')
FORGET = .93                 # per-observation decay of old evidence
PRIOR = 4.                   # pseudo-observations of weight at the parent value
LOW, HIGH = .3, 1.15


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _bin(ratio):
    return sum(ratio > e for e in EDGES)


class Calibrator:
    def __init__(self, data=None):
        self.cells = {}   # 'quality|bin' -> [weighted target sum, weight]
        self.count = 0
        try:
            for key, (s, w) in ((data or {}).get('cells') or {}).items():
                q, b = key.split('|')
                s, w = _num(s), _num(w)
                if q in QUALITIES and 0 <= int(b) <= len(EDGES) and s is not None and w is not None \
                        and s >= 0 and w >= 0:
                    self.cells[key] = [s, w]
            self.count = max(0, int((data or {}).get('count', 0)))
        except Exception:
            self.cells, self.count = {}, 0

    def to_dict(self):
        return {'cells': {k: list(v) for k, v in self.cells.items()}, 'count': self.count}

    def _pooled(self, qualities, upto):
        """Estimate from cells with bin <= upto, shrunk toward 1.0 (global) or its parent."""
        s = w = 0.
        for q in qualities:
            for b in range(upto + 1):
                cs, cw = self.cells.get('%s|%d' % (q, b), (0., 0.))
                s, w = s + cs, w + cw
        return s, w

    def _estimate(self, quality, b):
        gs, gw = self._pooled(QUALITIES, b)
        glob = (PRIOR + gs) / (PRIOR + gw)
        qs, qw = self._pooled((quality,), b)
        qual = (PRIOR * glob + qs) / (PRIOR + qw)
        cs, cw = self.cells.get('%s|%d' % (quality, b), (0., 0.))
        return (PRIOR * qual + cs) / (PRIOR + cw)

    def observe(self, check):
        try:
            quality = check['quality']
            price, ceiling = _num(check['price']), _num(check['value_ceiling'])
            expected, actual = _num(check['expected_demand']), _num(check['actual_sales'])
            meals = _num(check['meals'])
            if quality not in QUALITIES or None in (price, ceiling, expected, actual, meals):
                return
            if min(price, expected, actual, meals) < 0 or ceiling <= 0 or expected <= 0:
                return
            b = _bin(price / ceiling)
            target = min(actual / expected, 1.5)
            sold_out = bool(check.get('sellout')) or (meals > 0 and actual >= meals)
            # Decay first, then a sellout (lower bound) may only raise the estimate.
            for cell in self.cells.values():
                cell[0] *= FORGET
                cell[1] *= FORGET
            if sold_out and target <= self._estimate(quality, b):
                return
            cell = self.cells.setdefault('%s|%d' % (quality, b), [0., 0.])
            cell[0] += target
            cell[1] += 1.
            self.count += 1
        except Exception:
            return

    def factor(self, spot, quality, price, value_ceiling, weather=None):
        try:
            price, ceiling = _num(price), _num(value_ceiling)
            if quality not in QUALITIES or price is None or ceiling is None or ceiling <= 0 or price < 0:
                return 1.
            return min(HIGH, max(LOW, self._estimate(quality, _bin(price / ceiling))))
        except Exception:
            return 1.

    def report(self):
        out = {'observations': self.count, 'buckets': {}}
        for q in QUALITIES:
            for b in range(len(EDGES) + 1):
                cs, cw = self.cells.get('%s|%d' % (q, b), (0., 0.))
                out['buckets']['%s/r%d' % (q, b)] = {
                    'weight': round(cw, 2), 'factor': round(min(HIGH, max(LOW, self._estimate(q, b))), 3)}
        return out
