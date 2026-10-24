"""Food Truck Wars bot. Python 3.8+, standard library only.

    export NIGHTSHIFT_KEY=fk_your_key_here      (or put it in .env, which is gitignored)
    python3 bot.py                              # play live / final (leave it running)
    python3 bot.py --practice --seed 1 --rivals 3
    python3 bot.py --practice --seed 1 --dump state_sample.json   # save a state for inspection

Strategy (see decide()):
  * For every spot x quality x price, simulate tomorrow's customers one by one
    against the rivals we expect there (yesterday's trucks, plus a hedge that one
    rival migrates to yesterday's most profitable spot).
  * Stock is chosen by expected profit under a weather/noise demand mixture
    (weather.py), capped by the $500 budget.
  * Reputation earned today is valued by the extra pricing power it buys for the
    days left in the market, shrinking to zero at the end.
  * Customer price limits per spot self-calibrate from yesterday's results.
  * Any failure falls back to a safe valid order. The key is never printed.
"""
import argparse
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request

try:
    import weather
except Exception:          # bot must still run if the module is missing
    weather = None


def load_env(path=".env"):
    """Load KEY=value lines from a .env file next to this script. Real environment variables win."""
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

# ---------------------------------------------------------------- model constants

SPOTS = ["highst", "oval", "union", "northres", "thompson", "stadium"]
SHARE = {"highst": .24, "oval": .20, "union": .18, "northres": .16, "thompson": .14, "stadium": .08}
QUALS = ["cheap", "standard", "premium"]
QIDX = {"cheap": 0, "standard": 1, "premium": 2}
COSTS = {"cheap": 2.50, "standard": 3.75, "premium": 5.50}
QVAL = {"cheap": 8.0, "standard": 12.0, "premium": 16.0}
REP_STEP = {"cheap": -3.0, "standard": 2.0, "premium": 5.0}

# Prior customer price limits: (mean, sd).  Docs: North Res ~$11, High St wide
# spread, Stadium pickiest.  Means are re-calibrated from results each day.
PRIOR_LIMIT = {"highst": (13.0, 5.0), "oval": (12.5, 3.5), "union": (12.5, 3.5),
               "northres": (11.0, 3.0), "thompson": (12.5, 3.5), "stadium": (15.0, 4.0)}
# Prior share of customers whose minimum quality is cheap / standard / premium.
PRIOR_MINQ = {"stadium": (0.20, 0.40, 0.40)}
DEFAULT_MINQ = (0.45, 0.40, 0.15)

MIGRATE_P = 0.25        # chance one rival moves to yesterday's best spot
REP_WEIGHT = 0.0        # tested 0/.15/.3/.6/1.0 offline: 0 best (+$1.5k/market holdout)
REP_HORIZON = 12        # days of future benefit counted at most
RETURN_P = 0.6          # chance we come back to the spot we build rep at
SAMPLE_CUST = 120       # simulated customers per spot (scaled to forecast)
DEADLINE = 1.0          # seconds of compute before we stop searching

_limit_mu = {s: PRIOR_LIMIT[s][0] for s in SPOTS}   # calibrated means
_W = weather.WeatherModel.load() if weather else None
_market_key = None


# ---------------------------------------------------------------- helpers

def num(x, d=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else d
    except (TypeError, ValueError):
        return d


def get(d, *keys, default=None):
    """First present key among alternatives."""
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def rep_factor(rep):
    return 0.7 + 0.6 * max(0.0, min(100.0, rep)) / 100.0


def rules_of(state):
    r = state.get("rules") or {}
    costs = dict(COSTS)
    costs.update({k: num(v, COSTS.get(k, 5)) for k, v in (get(r, "costs", default={}) or {}).items()
                  if k in COSTS})
    qv = dict(QVAL)
    qv.update({k: num(v, QVAL.get(k, 10)) for k, v in
               (get(r, "quality_values", "values", default={}) or {}).items() if k in QVAL})
    spots = [s for s in (get(r, "spots", default=SPOTS) or SPOTS) if isinstance(s, str)] or SPOTS
    budget = num(get(r, "budget", default=500), 500)
    pmin = num(get(r, "min_price", "price_min", default=1), 1)
    pmax = num(get(r, "max_price", "price_max", default=30), 30)
    limits = get(r, "price_limits", "price_limit", default=None)
    return costs, qv, spots, budget, pmin, pmax, limits


def truck_records(last_day):
    """Yield (spot, record) for every truck line in last_day, any layout."""
    out = []
    if not isinstance(last_day, dict):
        return out
    spots = last_day.get("spots")
    if isinstance(spots, dict):
        for s, info in spots.items():
            for t in (get(info, "trucks", default=[]) or []) if isinstance(info, dict) else []:
                if isinstance(t, dict):
                    out.append((get(t, "spot", default=s), t))
    elif isinstance(spots, list):
        for info in spots:
            s = get(info, "spot", "name", "id")
            for t in (get(info, "trucks", default=[]) or []):
                if isinstance(t, dict):
                    out.append((get(t, "spot", default=s), t))
    for t in last_day.get("trucks") or []:
        if isinstance(t, dict) and t.get("spot"):
            out.append((t["spot"], t))
    return out


def spot_info(last_day, spot):
    spots = (last_day or {}).get("spots")
    if isinstance(spots, dict):
        return spots.get(spot) or {}
    if isinstance(spots, list):
        for info in spots:
            if get(info, "spot", "name", "id") == spot:
                return info
    return {}


def make_customers(rng, spot, n, mu):
    sd = PRIOR_LIMIT.get(spot, (12, 3.5))[1]
    mq = PRIOR_MINQ.get(spot, DEFAULT_MINQ)
    cust = []
    for _ in range(n):
        lim = max(0.5, rng.gauss(mu, sd))
        u = rng.random()
        q = 0 if u < mq[0] else (1 if u < mq[0] + mq[1] else 2)
        cust.append((lim, q))
    return cust


def simulate(cust, rivals, me):
    """Sequential customer choice. rivals: list of [value, price, qidx, stock].
    me: (value, price, qidx) with unlimited stock. Returns (my_sales, rival_sales)."""
    stock = [r[3] for r in rivals]
    mine = 0
    rsold = 0
    for lim, mq in cust:
        best_v, best_p, who = -1e9, 1e9, -2
        if me is not None:
            v, p, q = me
            if p <= lim and q >= mq and v >= 0:
                best_v, best_p, who = v, p, -1
        for i, r in enumerate(rivals):
            if stock[i] <= 0:
                continue
            v, p, q = r[0], r[1], r[2]
            if p <= lim and q >= mq and v >= 0 and (v > best_v or (v == best_v and p < best_p)):
                best_v, best_p, who = v, p, i
        if who == -1:
            mine += 1
        elif who >= 0:
            stock[who] -= 1
            rsold += 1
    return mine, rsold


# ---------------------------------------------------------------- calibration

def calibrate(state, costs, qv):
    """Nudge each spot's price-limit mean so a replay of yesterday matches sales."""
    ld = state.get("last_day") or {}
    recs = truck_records(ld)
    if not recs:
        return
    rng = random.Random(12345)
    by_spot = {}
    for s, t in recs:
        by_spot.setdefault(s, []).append(t)
    for s, trucks in by_spot.items():
        if s not in _limit_mu:
            continue
        info = spot_info(ld, s)
        n = int(num(get(info, "customers", "arrivals", "visitors"), 0))
        if n <= 0:
            continue
        actual = sum(num(get(t, "sold", "sales", "meals_sold"), 0) for t in trucks)
        k = min(n, SAMPLE_CUST)
        scale = k / float(n)
        rivals = []
        for t in trucks:
            q = get(t, "quality", default="standard")
            if q not in QIDX:
                continue
            p = num(get(t, "price"), 10)
            rep = num(get(t, "reputation", "rep"), 50)
            rivals.append([qv[q] * rep_factor(rep) - p, p, QIDX[q],
                           num(get(t, "meals", "stock"), 0) * scale])
        if not rivals:
            continue
        cust = make_customers(rng, s, k, _limit_mu[s])
        _, pred = simulate(cust, rivals, None)
        pred /= scale
        if actual > pred * 1.05 + 2:
            _limit_mu[s] = min(PRIOR_LIMIT[s][0] + 5, _limit_mu[s] + 0.4)
        elif actual < pred * 0.95 - 2:
            _limit_mu[s] = max(PRIOR_LIMIT[s][0] - 5, _limit_mu[s] - 0.4)


# ---------------------------------------------------------------- decision

def predict_rivals(state, spots, qv, my_name):
    """Expected rivals per spot as [value, price, qidx, meals], plus yesterday's best spot."""
    ld = state.get("last_day") or {}
    reps = {}
    for t in state.get("trucks") or []:
        if isinstance(t, dict):
            reps[get(t, "truck_name", "name")] = num(get(t, "reputation", "avg_reputation"), 50)
    per = {s: [] for s in spots}
    profit_per_truck = {}
    me = ld.get("you") if isinstance(ld.get("you"), dict) else {}
    my_sig = (me.get("spot"), num(me.get("price"), -1), me.get("quality"), num(me.get("meals"), -1))
    skipped_me = False
    for s, t in truck_records(ld):
        name = get(t, "truck_name", "name")
        sig = (s, num(t.get("price"), -2), t.get("quality"), num(get(t, "meals", "stock"), -2))
        if not skipped_me and not my_name and me and sig == my_sig:
            skipped_me = True
            continue
        if s not in per or (my_name and name == my_name) or t.get("you"):
            continue
        q = get(t, "quality", default="standard")
        if q not in QIDX:
            continue
        p = num(get(t, "price"), 10)
        rep = num(get(t, "reputation", "rep"), reps.get(name, 50))
        meals = num(get(t, "meals", "stock"), 133)
        per[s].append([qv[q] * rep_factor(rep) - p, p, QIDX[q], meals])
        prof = num(get(t, "profit"), num(get(t, "sold"), 0) * p - meals * COSTS[q])
        profit_per_truck.setdefault(s, []).append(prof)
    best = None
    if profit_per_truck:
        best = max(profit_per_truck, key=lambda s: sum(profit_per_truck[s]) / len(profit_per_truck[s]))
    if not any(per.values()):
        # day 1: assume starter-like rivals spread by spot share
        n = max(0, len(state.get("trucks") or []) - 1) or 3
        for s in spots:
            for _ in range(int(n * SHARE.get(s, 0.15) + 0.4)):
                per[s].append([qv["standard"] - 10.0, 10.0, 1, 133])
    return per, best


def remaining_days(state):
    day = num(state.get("order_for_day"), 1)
    total = num(get(state, "days", "total_days", "market_days", default=None), 0)
    if not total:
        total = num(get(state.get("rules") or {}, "days", "total_days", default=0), 0)
    if not total:
        total = 30 if state.get("mode") in ("practice", "final") else 48
    return max(0.0, total - day)


def fallback(state):
    try:
        costs = rules_of(state)[0]
    except Exception:
        costs = COSTS
    return {"spot": "highst", "price": 9.0, "quality": "standard",
            "meals": int(min(500, 500) // costs.get("standard", 3.75) * 0.7)}


def decide(state):
    global _market_key
    t0 = time.time()
    costs, qv, spots, budget, pmin, pmax, limits = rules_of(state)
    mkey = (state.get("mode"), state.get("market"))
    if mkey != _market_key:
        _market_key = mkey
        for sp in _limit_mu:                # fresh calibration each market (tested better)
            _limit_mu[sp] = PRIOR_LIMIT[sp][0]
        if isinstance(limits, dict):        # use official price-limit means if given
            for s, v in limits.items():
                m = num(v if not isinstance(v, dict) else get(v, "mean", "avg", "typical"), 0)
                if s in _limit_mu and m > 0:
                    _limit_mu[s] = m
    try:
        calibrate(state, costs, qv)
    except Exception:
        pass
    probs = None
    if _W is not None:
        try:
            _W.observe_last_day(state)
            probs = weather.forecast_probs(state)
            if num(state.get("order_for_day"), 0) % 5 == 0:
                _W.save()
        except Exception:
            probs = None

    you = state.get("you") or {}
    my_name = get(you, "truck_name", "name")
    my_rep = get(you, "reputation", default={}) or {}
    fc = state.get("forecast") or {}
    exp = get(fc, "expected_customers", default={}) or {}
    ntr = max(4, len(state.get("trucks") or []) or 4)
    rivals, best_spot = predict_rivals(state, spots, qv, my_name)
    rem = remaining_days(state)
    rng = random.Random(num(state.get("order_for_day"), 0) * 7919 + 17)

    # rank spots by forecast size so a deadline cut keeps the best candidates
    def ncust(s):
        n = num(exp.get(s) if isinstance(exp, dict) else None, 0)
        return n if n > 0 else 110.0 * ntr * SHARE.get(s, 0.15)
    order = sorted([s for s in spots if s in SHARE or True], key=lambda s: -ncust(s))

    best = None   # (ev, spot, q, price, meals)
    for s in order:
        n = ncust(s)
        if n <= 0:
            continue
        k = int(min(n, SAMPLE_CUST)) or 1
        scale = k / n
        cust = make_customers(rng, s, k, _limit_mu.get(s, 12.0))
        base = [[r[0], r[1], r[2], r[3] * scale] for r in rivals.get(s, [])]
        scen = [(1.0, base)]
        if best_spot and best_spot == s and MIGRATE_P > 0:
            # pick a rival from another spot; copy the strongest truck here
            src = max(base, key=lambda r: r[0]) if base else [qv["standard"] - 10, 10, 1, 133 * scale]
            scen = [(1 - MIGRATE_P, base), (MIGRATE_P, base + [list(src)])]
        rep = num(my_rep.get(s) if isinstance(my_rep, dict) else None, 50)
        mult = [(1.0, 1.0)]
        if _W is not None and probs:
            try:
                mult = _W.demand_scenarios(s, 1.0, probs, 3)
            except Exception:
                pass
        hi = min(pmax, _limit_mu.get(s, 12) + 2.5 * PRIOR_LIMIT.get(s, (12, 3.5))[1])
        for q in QUALS:
            c = costs[q]
            val0 = qv[q] * rep_factor(rep)
            p = max(pmin, math.floor(c) + 1.0)
            while p <= hi + 1e-9:
                if time.time() - t0 > DEADLINE:
                    break
                d = 0.0
                for w, rv in scen:
                    d += w * simulate(cust, rv, (val0 - p, p, QIDX[q]))[0]
                d /= scale                      # expected sales with unlimited stock
                meals, ev = best_meals(d, mult, p, c, budget)
                ev += rep_bonus(q, rep, min(d, meals), rem, qv)
                if best is None or ev > best[0]:
                    best = (ev, s, q, p, meals)
                p += 1.0
    if best is None:
        return fallback(state)
    # refine price around the winner in $0.25 steps
    ev, s, q, p0, _ = best
    try:
        n = ncust(s)
        k = int(min(n, SAMPLE_CUST)) or 1
        scale = k / n
        rep = num(my_rep.get(s) if isinstance(my_rep, dict) else None, 50)
        val0 = qv[q] * rep_factor(rep)
        mult = _W.demand_scenarios(s, 1.0, probs, 3) if (_W and probs) else [(1.0, 1.0)]
        base = [[r[0], r[1], r[2], r[3] * scale] for r in rivals.get(s, [])]
        for seed_off in (0,):
            cust = make_customers(random.Random(num(state.get("order_for_day"), 0) * 7919 + 17), s, k,
                                  _limit_mu.get(s, 12.0))
        for dp in (-0.75, -0.5, -0.25, 0.25, 0.5, 0.75):
            if time.time() - t0 > DEADLINE * 1.3:
                break
            p = round(p0 + dp, 2)
            if p < pmin or p > pmax:
                continue
            d = simulate(cust, base, (val0 - p, p, QIDX[q]))[0] / scale
            meals, e = best_meals(d, mult, p, costs[q], budget)
            e += rep_bonus(q, rep, min(d, meals), rem, qv)
            if e > best[0]:
                best = (e, s, q, p, meals)
    except Exception:
        pass
    ev, s, q, p, meals = best
    if ev <= 0 or meals <= 0:
        # nothing profitable: sit out cheaply rather than lose money
        return {"spot": s, "price": round(p, 2), "quality": q, "meals": 0}
    meals = int(min(meals, budget // costs[q]))
    return {"spot": s, "price": round(min(pmax, max(pmin, p)), 2), "quality": q, "meals": meals}


def best_meals(d, mult, price, cost, budget):
    """Expected-profit-maximising stock for mean demand d under multiplier mixture."""
    cap = int(budget // cost)
    if d <= 0 or price <= cost:
        return 0, 0.0
    scen = [(w, d * m) for w, m in mult]
    hi = min(cap, int(max(m for _, m in scen) * 1.0) + 2)
    best_q, best_ev = 0, 0.0
    for m in range(0, hi + 1):
        ev = price * sum(w * min(x, m) for w, x in scen) - cost * m
        if ev > best_ev:
            best_q, best_ev = m, ev
    return best_q, best_ev


def rep_bonus(q, rep, sales, rem, qv):
    """Value today's reputation change by future pricing power at this spot."""
    if sales <= 0 or rem <= 0:
        return 0.0
    new = max(0.0, min(100.0, rep + REP_STEP[q]))
    gain = (rep_factor(new) - rep_factor(rep)) * qv["premium"]   # $ of extra value per future sale
    return REP_WEIGHT * RETURN_P * gain * sales * min(rem, REP_HORIZON)


def safe_decide(state):
    try:
        order = decide(state)
        if not isinstance(order, dict) or order.get("spot") not in (rules_of(state)[2]):
            return fallback(state)
        return order
    except Exception as e:
        print("  decide failed (%s), using fallback" % type(e).__name__)
        return fallback(state)


# ---------------------------------------------------------------- plumbing

def call(method, path, body=None, timeout=35):
    """Returns (status, json). Network failures raise URLError/OSError/ValueError."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(URL + path, data=data, method=method, headers={
        "Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            try:
                return r.status, json.loads(raw.decode() or "{}")
            except ValueError:
                return r.status, {"error": "bad json"}
    except urllib.error.HTTPError as e:
        retry = e.headers.get("Retry-After") if e.headers else None
        try:
            body = json.load(e)
        except Exception:
            body = {"error": "http %d" % e.code}
        if not isinstance(body, dict):
            body = {"error": str(body)[:200]}
        if retry:
            body["_retry_after"] = retry
        return e.code, body


def _wait(res, default):
    try:
        return max(0.5, min(30.0, float(res.get("_retry_after", default))))
    except (TypeError, ValueError, AttributeError):
        return default


def play(base, dump=None):
    """Long-poll the state and send one order per night. Works for live, final and practice."""
    version, ordered_for, backoff = 0, None, 1
    while True:
        try:
            status, state = call("GET", base + "/state?since=%d" % version)
        except (urllib.error.URLError, OSError, ValueError) as e:
            print("network error (%s), retrying in %ds" % (type(e).__name__, backoff))
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue
        backoff = 1
        if status == 429:
            time.sleep(_wait(state, 2))
            continue
        if status != 200 or not isinstance(state, dict):
            print("error %d: %s" % (status, (state or {}).get("error") if isinstance(state, dict) else ""))
            if status in (401, 404) and base != "/api/v1":
                sys.exit(1)            # practice market gone / bad key
            if status == 401:
                sys.exit("bad key (401)")
            time.sleep(3)
            continue
        if dump:
            try:
                with open(dump, "w") as f:
                    json.dump(state, f, indent=1)
                print("  state saved to %s" % dump)
            except OSError:
                pass
            dump = None
        try:
            version = int(state.get("version", version))
        except (TypeError, ValueError):
            pass
        phase, mode = state.get("phase"), state.get("mode")
        if phase == "over" and mode == "practice":
            report(state)
            return state
        key = (mode, state.get("market"), state.get("order_for_day"))
        if phase == "night" and state.get("orders_open", True) and key != ordered_for:
            report(state)
            t0 = time.time()
            order = safe_decide(state)
            dt = time.time() - t0
            for attempt in range(3):
                try:
                    status, res = call("POST", base + "/orders", order, timeout=10)
                except (urllib.error.URLError, OSError, ValueError) as e:
                    print("  order network error (%s), retry" % type(e).__name__)
                    time.sleep(0.5 * (attempt + 1))
                    continue
                if status == 200:
                    ordered_for = key
                    print("  ordered for day %s (%.2fs): %s" % (res.get("for_day"), dt, order))
                    if "state" in res:          # practice plays the day right away
                        version = 0
                    break
                if status == 429:
                    time.sleep(_wait(res, 1))
                    continue
                print("  order rejected (%d): %s" % (status, res.get("error")))
                if status == 400:
                    order = fallback(state)     # bad input: try the safe order once
                    continue
                break
            else:
                time.sleep(1)
        elif phase not in ("night", "day", "over"):
            time.sleep(1)                       # unknown phase: avoid a tight loop


def report(state):
    try:
        last = state.get("last_day") or {}
        mine = last.get("you") or {}
        you = state.get("you") or {}
        where = str(state.get("mode")) + (" market %s" % state.get("market") if state.get("mode") == "final" else "")
        if mine.get("spot"):
            print("[%s] day %s: %s at %s $%s, sold %s/%s, profit $%.2f, cash $%.2f" % (
                where, last.get("day"), mine.get("quality"), mine.get("spot"), mine.get("price"),
                mine.get("sold"), mine.get("meals"), num(mine.get("profit")), num(you.get("cash"))))
        elif last.get("day"):
            print("[%s] day %s: sat out, cash $%.2f" % (where, last.get("day"), num(you.get("cash"))))
        else:
            print("[%s] night before day %s, cash $%.2f" % (where, state.get("order_for_day"), num(you.get("cash"))))
    except Exception:
        pass


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Food Truck Wars bot")
    ap.add_argument("--practice", action="store_true", help="play a private practice market")
    ap.add_argument("--seed", help="practice seed, for repeatable markets")
    ap.add_argument("--days", type=int, default=30, help="practice market length")
    ap.add_argument("--rivals", type=int, help="number of house rivals in practice")
    ap.add_argument("--dump", help="save the first state seen to this JSON file")
    args = ap.parse_args()
    if not KEY:
        sys.exit("Set NIGHTSHIFT_KEY first (you get it when you register).")
    if args.practice:
        body = {"days": args.days}
        if args.seed:
            body["seed"] = args.seed
        if args.rivals is not None:
            body["rivals"] = args.rivals
        status, res = call("POST", "/api/v1/practice", body)
        if status not in (200, 201):
            sys.exit("could not start practice (%d): %s" % (status, res.get("error")))
        print("practice market %s started" % res["id"])
        final = play(res["base"], args.dump)
        you = final.get("you") or {}
        print("practice over: cash $%.2f" % num(you.get("cash")))
        for t in sorted(final.get("trucks") or [], key=lambda t: -num(t.get("cash"))):
            print("  %-24s $%10.2f" % (t.get("truck_name"), num(t.get("cash"))))
    else:
        print("playing the live game at %s (Ctrl+C to stop)" % URL)
        while True:            # never exit on a recoverable error
            try:
                play("/api/v1")
            except KeyboardInterrupt:
                raise
            except SystemExit:
                raise
            except Exception as e:
                print("loop crashed (%s), restarting in 3s" % type(e).__name__)
                time.sleep(3)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
