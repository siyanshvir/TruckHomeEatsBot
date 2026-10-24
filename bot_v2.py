"""Food Truck Wars bot v2 -- alternative algorithm. Python 3.8+, standard library only.

    export NIGHTSHIFT_KEY=...            # or put it in a .env file (gitignored)
    python3 bot_v2.py                    # live / final
    python3 bot_v2.py --practice --seed 7 --rivals 5

Approach: for every (spot, quality, price) it computes our expected uncapped demand with a
value-ordered "cascade" (higher-value rivals absorb eligible customers up to their stock,
the overflow spills to us), averages two rival scenarios (rivals stay put / one extra
rival crowds into yesterday's busiest spot), stocks meals by the newsvendor critical
fractile, and adds a reputation bonus for future pricing power. Rival per-spot reputation
is reconstructed from daily history. Our demand model self-calibrates per spot from
uncensored days (days we did not sell out).
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request


SPOTS = ["oval", "thompson", "union", "stadium", "highst", "northres"]
QV = {"cheap": 8.0, "standard": 12.0, "premium": 16.0}
QR = {"cheap": 0, "standard": 1, "premium": 2}
COST = {"cheap": 2.5, "standard": 3.75, "premium": 5.5}
REP_DELTA = {"cheap": -3.0, "standard": 2.0, "premium": 5.0}
SHARE = {"highst": .24, "oval": .20, "union": .18, "northres": .16, "thompson": .14, "stadium": .08}

# Customer priors per spot from the official docs: (limit mean, limit sd, P(min=cheap), P(min=premium)).
# Calibrated online from observed sales where possible (see Memory.calibrate).
POP = {
    "highst":   (12.0, 3.0, 0.50, 0.12),
    "oval":     (13.0, 2.2, 0.35, 0.20),
    "union":    (13.0, 2.2, 0.20, 0.22),
    "northres": (11.0, 2.0, 0.48, 0.07),
    "thompson": (12.0, 2.2, 0.45, 0.15),
    "stadium":  (13.0, 2.2, 0.35, 0.25),
}
# Event tweaks to the population (limit shift, cheap-share shift, premium-share shift)
EVENT_POP = {
    "game": (0.8, 0.15, -0.05), "midterm": (0.0, -0.10, 0.15), "concert": (0.8, 0.0, 0.0),
    "late": (0.0, 0.20, -0.05), "rally": (0.0, 0.0, 0.0),
}

PARAMS = {
    "demand_sd": 0.12,      # relative sd of actual customers vs forecast
    "hedge": 0.5,          # weight on "rivals migrate toward yesterday's best spot" scenario
    "rep_value": 3.0,       # $ value per rep point per remaining day used at that spot (scaled below)
    "rep_horizon": 12,      # days of future use we credit for reputation
    "stick": 1.0,           # probability we return to a spot (scales rep value)
    "price_step": 0.25,
    "k_quant": 16,
}

_Z = None


def _zs(k):
    global _Z
    if _Z is None or len(_Z) != k:
        out = []
        for i in range(k):
            p = (i + 0.5) / k
            lo, hi = -6.0, 6.0
            for _ in range(40):
                mid = (lo + hi) / 2
                if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
                    lo = mid
                else:
                    hi = mid
            out.append((lo + hi) / 2)
        _Z = out
    return _Z


def rf(rep):
    return 0.7 + 0.6 * rep / 100.0


def population(spot, events):
    mu, sd, pc, pp = POP.get(spot, (12.0, 2.5, 0.4, 0.15))
    for ev in events or []:
        if ev.get("spot") not in (spot, None, "all", "campus") and ev.get("_kind") not in ("game", "late"):
            continue
        k = ev.get("_kind")
        if k in EVENT_POP and (ev.get("spot") == spot or k in ("game", "late")):
            a, b, c = EVENT_POP[k]
            if k == "game" and ev.get("spot") not in (spot, "stadium") and spot != "stadium":
                a, b, c = a * 0.5, b * 0.5, c * 0.5
            mu += a
            pc += b
            pp += c
    pc = min(max(pc, 0.0), 0.9)
    pp = min(max(pp, 0.0), 0.9 - pc)
    ps = 1.0 - pc - pp
    cells = []
    zs = _zs(PARAMS["k_quant"])
    k = len(zs)
    for z in zs:
        lim = mu + sd * z
        cells.append((lim, 0, pc / k))
        cells.append((lim, 1, ps / k))
        cells.append((lim, 2, pp / k))
    return cells


def event_kind(name):
    n = (name or "").lower()
    if "game" in n or "football" in n:
        return "game"
    if "midterm" in n or "exam" in n:
        return "midterm"
    if "concert" in n:
        return "concert"
    if "late" in n or "night" in n or "rush" in n:
        return "late"
    if "rally" in n:
        return "rally"
    return None


def my_demand(count, cells, me, rivals):
    """Expected customers who would buy from `me` (uncapped) given capped rivals.

    me: (price, quality, rep). rivals: list of (price, quality, rep, meals).
    Sequential arrival approximated: customers rank eligible trucks by value; a
    rival that is over-subscribed passes its excess (proportionally) down the list.
    """
    mp, mq, mr = me
    mv = QV[mq] * rf(mr) - mp
    if mv < 0:
        return 0.0
    mrank = QR[mq]
    rv = []
    for (p, q, r, m) in rivals:
        rv.append((QV[q] * rf(r) - p, p, QR[q], float(m)))
    # rivals strictly preferred over us (higher value, or tie and cheaper)
    better = [i for i, (v, p, qr, m) in enumerate(rv) if v > mv + 1e-9 or (abs(v - mv) <= 1e-9 and p < mp)]
    better.sort(key=lambda i: (-rv[i][0], rv[i][1]))
    # customers eligible for us
    total = 0.0
    if not better:
        for lim, mn, w in cells:
            if mp <= lim and mrank >= mn:
                total += w
        return total * count
    # Fill better rivals in value order with customers eligible for them; leftover eligible-for-us go to us.
    # Compute per-cell mass; each better rival absorbs from cells eligible for it until capacity.
    mass = [w * count for (_, _, w) in cells]
    for i in better:
        v, p, qr, cap = rv[i]
        if v < 0:
            continue
        elig = [ci for ci, (lim, mn, _) in enumerate(cells) if p <= lim and qr >= mn]
        tot = sum(mass[ci] for ci in elig)
        if tot <= 0:
            continue
        take = min(1.0, cap / tot)
        for ci in elig:
            mass[ci] *= (1 - take)
    for ci, (lim, mn, _) in enumerate(cells):
        if mp <= lim and mrank >= mn:
            total += mass[ci]
    return total


def exp_min(d, m, sd):
    """E[min(d*X, m)] with X ~ N(1, sd) truncated at 0 (closed form approx)."""
    if d <= 0:
        return 0.0
    s = max(d * sd, 1e-6)
    z = (m - d) / s
    phi = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    # E[min(D,m)] = d - E[(D-m)+]; E[(D-m)+] = s*phi + (d-m)*(1-Phi)
    return max(0.0, d - (s * phi + (d - m) * (1 - Phi)))


def best_meals(d, price, cost, budget, sd):
    if d <= 0 or price <= cost:
        return 0, 0.0
    maxm = int(budget // cost)
    crit = (price - cost) / price
    # newsvendor quantile of normal
    lo, hi = -5.0, 5.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < crit:
            lo = mid
        else:
            hi = mid
    m = int(round(d * (1 + sd * (lo + hi) / 2)))
    m = max(0, min(maxm, m))
    best = (m, price * exp_min(d, m, sd) - cost * m)
    for mm in (m - 2, m + 2):
        if 0 <= mm <= maxm:
            v = price * exp_min(d, mm, sd) - cost * mm
            if v > best[1]:
                best = (mm, v)
    return best


class Memory:
    def __init__(self):
        self.key = None
        self.rival_rep = {}      # truck id -> {spot: rep}
        self.seen_day = 0
        self.history = []        # last_day dicts
        self.calib = {s: [0.0, 0.0] for s in SPOTS}  # predicted vs actual our sales (ratio tracker)
        self.last_pred = None

    def reset(self, key):
        self.__init__()
        self.key = key


MEM = Memory()


def _update_memory(state):
    key = (state.get("mode"), state.get("market"))
    if MEM.key != key:
        MEM.reset(key)
    ld = state.get("last_day") or {}
    day = ld.get("day") or 0
    if not day or day <= MEM.seen_day:
        return
    MEM.seen_day = day
    MEM.history.append(ld)
    MEM.history = MEM.history[-6:]
    me_id = (state.get("you") or {}).get("id")
    spots = ld.get("spots") or {}
    visited = {}
    for s, info in spots.items():
        for t in (info or {}).get("trucks") or []:
            tid = t.get("id")
            if tid is None or tid == me_id:
                continue
            reps = MEM.rival_rep.setdefault(tid, {x: 50.0 for x in SPOTS})
            visited.setdefault(tid, set()).add(s)
            if (t.get("sold") or 0) > 0 and t.get("quality") in REP_DELTA:
                reps[s] = min(100.0, max(0.0, reps.get(s, 50.0) + REP_DELTA[t["quality"]]))
    for tid, reps in MEM.rival_rep.items():
        for s in SPOTS:
            if s not in visited.get(tid, set()):
                reps[s] = reps.get(s, 50.0) + (50.0 - reps.get(s, 50.0)) * 0.03
    # calibration of our own demand model: compare predicted vs actual when we did not sell out
    mine = ld.get("you") or {}
    if MEM.last_pred and mine.get("spot") == MEM.last_pred[0] and mine.get("meals"):
        sold, meals = mine.get("sold") or 0, mine.get("meals") or 0
        pred = MEM.last_pred[1]
        if sold < meals and pred > 1:  # uncensored
            c = MEM.calib[mine["spot"]]
            c[0] += pred
            c[1] += sold


def _rival_offers(state):
    """Return scenarios: list of (weight, {spot: [(price, quality, rep, meals)]})."""
    ld = state.get("last_day") or {}
    me_id = (state.get("you") or {}).get("id")
    spots = ld.get("spots") or {}
    stay = {s: [] for s in SPOTS}
    rivals = []
    best_spot, best_spt = None, -1.0
    for s, info in spots.items():
        trucks = [t for t in (info or {}).get("trucks") or [] if t.get("id") != me_id]
        n_all = len((info or {}).get("trucks") or [])
        cust = (info or {}).get("customers") or 0
        per = cust / max(1, n_all)
        if per > best_spt:
            best_spot, best_spt = s, per
        for t in trucks:
            q = t.get("quality") if t.get("quality") in QV else "standard"
            try:
                p = float(t.get("price") or 10)
                m = float(t.get("meals") or 0)
            except (TypeError, ValueError):
                continue
            rep = MEM.rival_rep.get(t.get("id"), {}).get(s, 50.0)
            off = (p, q, rep + REP_DELTA[q] if (t.get("sold") or 0) > 0 else rep, m)
            rivals.append((t.get("id"), s, off))
            if s in stay:
                stay[s].append(off)
    if not rivals:
        return [(1.0, stay)], 0
    h = PARAMS["hedge"]
    mig = {s: [] for s in SPOTS}
    for tid, s, off in rivals:
        mig[s].append(off)
    if best_spot in mig:
        # one extra rival copy appears at yesterday's best spot (crowding risk)
        p, q, r, m = sorted([o for _, _, o in rivals], key=lambda o: o[0])[len(rivals) // 2]
        mig[best_spot] = mig[best_spot] + [(p, q, 50.0, m)]
    return [(1 - h, stay), (h, mig)], len(rivals)


def plan(state):
    t0 = time.time()
    _update_memory(state)
    rules = state.get("rules") or {}
    budget = float(rules.get("budget") or 500)
    costs = dict(COST)
    costs.update({k: float(v) for k, v in (rules.get("costs") or {}).items() if k in COST})
    pmin = float(rules.get("price_min") or 1)
    pmax = float(rules.get("price_max") or 30)
    fc = state.get("forecast") or {}
    exp = fc.get("expected_customers") or {}
    events = []
    for ev in fc.get("events") or []:
        if isinstance(ev, dict):
            e = dict(ev)
            e["_kind"] = event_kind(ev.get("name"))
            events.append(e)
    myrep = (state.get("you") or {}).get("reputation") or {}
    day = state.get("order_for_day") or 1
    days = state.get("days") or 30
    remaining = max(0, days - day)  # days after this one
    scen, nriv = _rival_offers(state)
    ntrucks = nriv + 1
    sd = PARAMS["demand_sd"]
    hz = min(PARAMS["rep_horizon"], remaining)
    best = None
    spots = [s for s in (rules.get("spots") or SPOTS) if s in SPOTS] or SPOTS
    for s in spots:
        count = exp.get(s)
        if not isinstance(count, (int, float)) or count <= 0:
            count = 110.0 * max(4, ntrucks) * SHARE.get(s, 0.15)
        c = MEM.calib.get(s)
        if c and c[0] > 30:
            count *= min(1.4, max(0.6, c[1] / c[0]))
        cells = population(s, [e for e in events])
        rep = float(myrep.get(s, 50.0))
        for q in ("premium", "standard", "cheap"):
            cost = costs[q]
            # rep value: future pricing power. Each rep point raises value by QV*0.006 ~ 0.1 $/meal
            dr = max(-rep, min(100 - rep, REP_DELTA[q]))
            rep_bonus = dr * PARAMS["rep_value"] * hz * PARAMS["stick"]
            maxp = min(pmax, QV[q] * rf(rep))
            p = max(pmin, cost + 0.5)
            while p <= maxp + 1e-9:
                ev = 0.0
                d_mix = 0.0
                for w, offers in scen:
                    d = my_demand(count, cells, (p, q, rep), offers.get(s, []))
                    d_mix += w * d
                m, prof = best_meals(d_mix, p, cost, budget, sd)
                if m > 0:
                    # evaluate meals across scenarios
                    ev = 0.0
                    for w, offers in scen:
                        d = my_demand(count, cells, (p, q, rep), offers.get(s, []))
                        ev += w * (p * exp_min(d, m, sd) - cost * m)
                    score = ev + (rep_bonus if ev > 0 else 0.0)
                    if best is None or score > best[0]:
                        best = (score, s, p, q, m, ev, d_mix)
                p += PARAMS["price_step"]
    if best is None:
        return None
    score, s, p, q, m, ev, d = best
    MEM.last_pred = (s, d)
    return {"spot": s, "price": round(p, 2), "quality": q, "meals": int(m)}, ev, time.time() - t0


def fallback(state):
    rules = (state or {}).get("rules") or {}
    spots = rules.get("spots") or SPOTS
    spot = "union" if "union" in spots else spots[0]
    return {"spot": spot, "price": 11.0, "quality": "standard", "meals": 110}


def decide(state):
    try:
        r = plan(state)
        if r and r[0]["meals"] >= 0:
            return r[0]
    except Exception as e:  # never let strategy errors cost an order
        print("  strategy error, using fallback: %r" % (e,))
    return fallback(state)


# ---------------------------------------------------------------- plumbing

def load_env(path=".env"):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                name, _, value = line.partition("=")
                value = value.strip().strip("'\"")
                if value != "fk_your_key_here":
                    os.environ.setdefault(name.strip(), value)
    except OSError:
        pass


load_env()
URL = os.environ.get("NIGHTSHIFT_URL", "https://nightshift.joinbuilders.org").rstrip("/")
KEY = os.environ.get("NIGHTSHIFT_KEY", "")


def call(method, path, body=None, timeout=35):
    """Returns (status, json_dict, retry_after_seconds). Never raises for HTTP/JSON problems."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(URL + path, data=data, method=method, headers={
        "Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            try:
                js = json.loads(raw.decode() or "{}")
            except ValueError:
                return 599, {"error": "malformed response"}, None
            return r.status, js if isinstance(js, dict) else {"error": "unexpected response"}, None
    except urllib.error.HTTPError as e:
        ra = None
        try:
            ra = float(e.headers.get("Retry-After")) if e.headers and e.headers.get("Retry-After") else None
        except (TypeError, ValueError):
            ra = None
        try:
            js = json.loads(e.read().decode() or "{}")
        except Exception:
            js = {}
        if not isinstance(js, dict):
            js = {}
        js.setdefault("error", "HTTP %d" % e.code)
        return e.code, js, ra


def post_order(base, order):
    """Up to 3 quick attempts; the night lasts minutes but the final waits only ~2s per day."""
    for i in range(3):
        try:
            status, res, ra = call("POST", base + "/orders", order, timeout=8)
        except (urllib.error.URLError, OSError) as e:
            print("  order network error (%s), retry" % e.__class__.__name__)
            time.sleep(0.3 * (i + 1))
            continue
        if status == 429:
            time.sleep(min(ra or 1.0, 3.0))
            continue
        return status, res
    return 0, {"error": "order failed after retries"}


def play(base):
    """Long-poll the state and send one order per night. Works for live, final and practice."""
    version, ordered_for, backoff = 0, None, 1.0
    while True:
        try:
            status, state, ra = call("GET", base + "/state?since=%d" % version)
        except (urllib.error.URLError, OSError) as e:
            print("network error (%s), retrying in %.0fs" % (e.__class__.__name__, backoff))
            time.sleep(backoff)
            backoff = min(backoff * 2, 20)
            continue
        except Exception as e:  # anything else: never die
            print("unexpected error (%r), retrying" % (e,))
            time.sleep(backoff)
            backoff = min(backoff * 2, 20)
            continue
        if status == 429:
            time.sleep(min(max(ra or 2.0, 0.5), 30))
            continue
        if status != 200:
            print("error %d: %s" % (status, state.get("error")))
            if status in (401, 403, 404):
                if base != "/api/v1":
                    sys.exit(1)
                time.sleep(10)        # live/final: keep trying, the organizers may fix it
                continue
            time.sleep(backoff)
            backoff = min(backoff * 2, 20)
            continue
        backoff = 1.0
        try:
            version = int(state.get("version") or 0)
        except (TypeError, ValueError):
            version = 0
        phase = state.get("phase")
        if phase == "over" and state.get("mode") == "practice":
            report(state)
            return state
        key = (state.get("mode"), state.get("market"), state.get("order_for_day"))
        if phase == "night" and state.get("orders_open", True) and key != ordered_for:
            report(state)
            t0 = time.time()
            order = decide(state)
            dt = time.time() - t0
            status, res = post_order(base, order)
            if status == 200:
                ordered_for = key
                print("  ordered for day %s: %s (%.0f ms)" % (res.get("for_day"), order, dt * 1000))
                if "state" in res:          # practice plays the day right away
                    version = 0
            else:
                print("  order rejected (%d): %s" % (status, res.get("error")))
                if status == 400:          # invalid order: send the safe fallback right away
                    status, res = post_order(base, fallback(state))
                    if status == 200:
                        ordered_for = key
                time.sleep(0.5)
        elif phase == "over":
            time.sleep(5)


def report(state):
    try:
        last = state.get("last_day") or {}
        mine = last.get("you") or {}
        mode = state.get("mode")
        where = "%s" % mode + (" market %s" % state.get("market") if mode == "final" else "")
        cash = (state.get("you") or {}).get("cash", 0) or 0
        if mine.get("spot"):
            print("[%s] day %s: %s at %s $%s, sold %s/%s, profit $%.2f, cash $%.2f" % (
                where, last.get("day"), mine.get("quality"), mine.get("spot"), mine.get("price"),
                mine.get("sold"), mine.get("meals"), mine.get("profit", 0) or 0, cash))
        else:
            print("[%s] night before day %s, cash $%.2f" % (where, state.get("order_for_day"), cash))
    except Exception:
        pass


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Food Truck Wars bot v2")
    ap.add_argument("--practice", action="store_true", help="play a private practice market")
    ap.add_argument("--seed", help="practice seed, for repeatable markets")
    ap.add_argument("--days", type=int, default=30, help="practice market length")
    ap.add_argument("--rivals", type=int, help="practice house bots (default 3)")
    args = ap.parse_args()
    if not KEY:
        sys.exit("Set NIGHTSHIFT_KEY first (environment variable or .env file).")
    if args.practice:
        body = {"days": args.days}
        if args.seed:
            body["seed"] = args.seed
        if args.rivals is not None:
            body["rivals"] = args.rivals
        status, res, _ = call("POST", "/api/v1/practice", body)
        if status not in (200, 201):
            sys.exit("could not start practice (%d): %s" % (status, res.get("error")))
        print("practice market %s started" % res.get("id"))
        final = play(res["base"])
        print("practice over: cash $%.2f" % final["you"]["cash"])
        for t in sorted(final.get("trucks") or [], key=lambda t: -(t.get("cash") or 0)):
            print("  %-24s $%10.2f" % (t.get("truck_name"), t.get("cash") or 0))
    else:
        print("playing the live game at %s (Ctrl+C to stop)" % URL)
        play("/api/v1")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
