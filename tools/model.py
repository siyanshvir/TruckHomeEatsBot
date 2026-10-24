"""Customer-choice model shared by the fitter and the bot's planner (kept dependency free)."""
import math

QV = {"cheap": 8.0, "standard": 12.0, "premium": 16.0}
QR = {"cheap": 0, "standard": 1, "premium": 2}
K = 24  # limit quantiles


def _norm_ppf(p):
    # Acklam-free: use bisection on erf, fine for K small
    lo, hi = -6.0, 6.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


_Z = [_norm_ppf((i + 0.5) / K) for i in range(K)]


def population(mu, sd, pc, pp):
    """Return list of (limit, minq_rank, weight fraction)."""
    ps = max(0.0, 1.0 - pc - pp)
    mins = [(0, pc), (1, ps), (2, pp)]
    cells = []
    for z in _Z:
        lim = mu + sd * z
        for r, p in mins:
            if p > 0:
                cells.append((lim, r, p / K))
    return cells


def rep_factor(rep):
    return 0.7 + 0.6 * rep / 100.0


def expected_sold(count, cells, offers):
    """offers: list of dict(price, quality, meals, rep). Returns expected sold per offer."""
    n = len(offers)
    vals = [QV[o["quality"]] * rep_factor(o["rep"]) - o["price"] for o in offers]
    rank = [QR[o["quality"]] for o in offers]
    cap = [float(o["meals"]) for o in offers]
    sold = [0.0] * n
    sold_out = [False] * n
    w = [c[2] * count for c in cells]
    for _ in range(n + 1):
        demand = [0.0] * n
        pick = []
        for ci, (lim, mr, _) in enumerate(cells):
            best, bv = -1, 0.0
            for i in range(n):
                if sold_out[i] or vals[i] <= 0 or o_price(offers, i) > lim or rank[i] < mr:
                    continue
                v = vals[i]
                if best < 0 or v > bv + 1e-12 or (abs(v - bv) <= 1e-12 and offers[i]["price"] < offers[best]["price"]):
                    best, bv = i, v
            pick.append(best)
            if best >= 0:
                demand[best] += w[ci]
        over = [i for i in range(n) if not sold_out[i] and demand[i] > cap[i] - sold[i] + 1e-9]
        if not over:
            for i in range(n):
                sold[i] += demand[i]
            break
        frac = {}
        for i in over:
            room = cap[i] - sold[i]
            frac[i] = room / demand[i] if demand[i] > 0 else 0
            sold[i] += room
            sold_out[i] = True
        for i in range(n):
            if i not in frac:
                sold[i] += demand[i] * 0  # recomputed next round
        new_w = []
        for ci, b in enumerate(pick):
            if b in frac:
                new_w.append(w[ci] * (1 - frac[b]))      # unserved, re-choose
            else:
                new_w.append(0.0)                         # others recomputed below
        # customers who chose a non-overfull truck stay with it; add their sales now
        for ci, b in enumerate(pick):
            if b >= 0 and b not in frac:
                sold[b] += w[ci]
        w = new_w
    return sold


def o_price(offers, i):
    return offers[i]["price"]
