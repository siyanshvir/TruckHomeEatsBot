"""Night Shift food truck bot: customer-choice model + newsvendor sizing.
Python 3.8+, standard library only.

    export NIGHTSHIFT_KEY=fk_your_key_here
    python3 truck_bot.py                 # live game / final (leave it running all night)
    python3 truck_bot.py --practice      # one practice market
    python3 truck_bot.py --bench 10      # 10 practice markets, prints average profit
    python3 truck_bot.py --dump          # also saves each night's raw state to last_state.json

How it decides, every night:
  1. Tracks every rival's reputation at every spot by replaying the reputation rules on last_day.
  2. Assumes rivals park roughly where they parked yesterday (plus "somebody will notice" filler
     trucks at crowded spots).
  3. Simulates 400 customers per spot (price limit + minimum quality), and for every
     spot x quality x price works out how many would pick us, plus spillover when rivals sell out.
  4. Cooks the profit-maximizing number of meals for an uncertain crowd (newsvendor), so we stop
     throwing away 100 meals a day.
  5. Adds a small bonus for building reputation where we'll keep selling, then picks the best.
  6. Self-calibrates its demand estimate from its own sales each day.
"""
import argparse
import json
import math
import os
import random
import sys
import time
import traceback
import urllib.error
import urllib.request
from statistics import NormalDist

URL = os.environ.get("NIGHTSHIFT_URL", "https://nightshift.joinbuilders.org").rstrip("/")
KEY = os.environ.get("NIGHTSHIFT_KEY", "")

SPOTS = ("highst", "oval", "union", "northres", "thompson", "stadium")
QUALS = ("cheap", "standard", "premium")
QV = {"cheap": 8.0, "standard": 12.0, "premium": 16.0}
COST = {"cheap": 2.50, "standard": 3.75, "premium": 5.50}
RDELTA = {"cheap": -3, "standard": 2, "premium": 5}
QRANK = {"cheap": 0, "standard": 1, "premium": 2}
SHARE = {"highst": .24, "oval": .20, "union": .18, "northres": .16, "thompson": .14, "stadium": .08}

# Customer model per spot: price-limit mean/sd, and share whose minimum quality is cheap/standard/premium.
BASE = {
    "highst":   {"mu": 12.0, "sd": 3.0, "mix": (0.50, 0.35, 0.15)},
    "oval":     {"mu": 13.0, "sd": 2.0, "mix": (0.35, 0.45, 0.20)},
    "union":    {"mu": 13.0, "sd": 2.0, "mix": (0.20, 0.60, 0.20)},
    "northres": {"mu": 11.0, "sd": 2.0, "mix": (0.55, 0.40, 0.05)},
    "thompson": {"mu": 12.0, "sd": 2.0, "mix": (0.45, 0.40, 0.15)},
    "stadium":  {"mu": 13.0, "sd": 2.2, "mix": (0.30, 0.45, 0.25)},
}

N_SAMPLES = 400
SIGMA = 0.30                       # demand uncertainty (crowd noise + model error), lognormal
ZS = [NormalDist().inv_cdf((i + 0.5) / 21) for i in range(21)]
PRICES = [round(5.0 + 0.25 * i, 2) for i in range(57)]   # $5.00 .. $19.00
DUMP = False
QUIET = False


def log(*a):
    if not QUIET:
        print(*a)


def num(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def rf(rep):
    return 0.7 + 0.6 * rep / 100.0


# ---------------------------------------------------------------- reading the state defensively

def spot_entries(last):
    sp = (last or {}).get("spots") or {}
    out = {}
    if isinstance(sp, dict):
        for s, v in sp.items():
            if isinstance(v, dict):
                out[s] = v
    elif isinstance(sp, list):
        for v in sp:
            if isinstance(v, dict):
                s = v.get("spot") or v.get("id") or v.get("name")
                out[s] = v
    return out


def offers_in(v):
    lst = v.get("trucks") or v.get("offers") or v.get("sellers") or []
    if isinstance(lst, dict):
        lst = [dict(x, truck_name=k) for k, x in lst.items() if isinstance(x, dict)]
    return [o for o in lst if isinstance(o, dict)]


def tname(o):
    return str(o.get("truck_name") or o.get("name") or o.get("truck") or o.get("truck_id") or o.get("id") or "?")


def sold_of(o):
    return num(o.get("sold", o.get("sales", o.get("meals_sold", 0))))


def my_name(state):
    you = state.get("you") or {}
    n = you.get("truck_name") or you.get("name") or you.get("truck")
    return str(n) if n else None


def is_me(o, spot, state):
    me = my_name(state)
    if me and tname(o) == me:
        return True
    mine = (state.get("last_day") or {}).get("you") or {}
    return (mine.get("spot") == spot and abs(num(o.get("price")) - num(mine.get("price"), -1)) < 1e-6
            and num(o.get("meals")) == num(mine.get("meals"), -1) and o.get("quality") == mine.get("quality"))


def events_text(state):
    fc = state.get("forecast") or {}
    try:
        return json.dumps(fc.get("events") or []).lower()
    except Exception:
        return ""


# ---------------------------------------------------------------- memory across nights

class Brain:
    def __init__(self):
        self.reset(None)

    def reset(self, key):
        self.key = key
        self.rep = {}        # rival name -> spot -> estimated reputation
        self.seen = set()    # days already ingested
        self.calib = 1.0     # demand multiplier learned from our own sales
        self.pred = {}       # day -> predicted demand for our order
        self.last_ofd = 0


BRAIN = Brain()


def ingest(state):
    key = (state.get("mode"), state.get("market"))
    ofd = int(num(state.get("order_for_day"), 0))
    if key != BRAIN.key or ofd < BRAIN.last_ofd:
        BRAIN.reset(key)
    BRAIN.last_ofd = ofd

    last = state.get("last_day") or {}
    day = last.get("day")
    if not day or day in BRAIN.seen:
        return
    BRAIN.seen.add(day)

    visited = {}
    for s, v in spot_entries(last).items():
        for o in offers_in(v):
            if not is_me(o, s, state):
                visited[tname(o)] = (s, o.get("quality"), sold_of(o))
    names = set(visited) | {tname(t) for t in (state.get("trucks") or []) if isinstance(t, dict)}
    names.discard(my_name(state) or "")
    for n in names:
        reps = BRAIN.rep.setdefault(n, {s: 50.0 for s in SPOTS})
        vis = visited.get(n)
        for s in SPOTS:
            if vis and vis[0] == s:
                if vis[2] > 0:
                    reps[s] = clamp(reps[s] + RDELTA.get(vis[1], 0), 0, 100)
            else:
                reps[s] += (50.0 - reps[s]) * 0.03

    # self-calibration from our own result
    mine = last.get("you") or {}
    pred = BRAIN.pred.get(day)
    meals = num(mine.get("meals"))
    if pred and pred > 1 and meals > 0:
        sold = num(mine.get("sold"))
        r = None
        if sold < meals - 0.5:
            r = sold / pred                 # not sold out: true demand observed
        elif pred < meals:
            r = meals / pred                # sold out: demand was at least this
        if r is not None:
            r = clamp(r, 0.3, 3.0)
            BRAIN.calib = clamp(math.exp(0.7 * math.log(BRAIN.calib) + 0.3 * math.log(r)), 0.5, 1.8)


def rival_rep(name, spot, state):
    if name in BRAIN.rep:
        return BRAIN.rep[name].get(spot, 50.0)
    for t in state.get("trucks") or []:
        if isinstance(t, dict) and tname(t) == name:
            r = t.get("reputation", t.get("avg_reputation"))
            if isinstance(r, dict):
                return num(r.get(spot), 50.0)
            return num(r, 50.0)
    return 50.0


# ---------------------------------------------------------------- the model

def spot_params(spot, ev, rules):
    p = dict(BASE[spot])
    mix = list(p["mix"])
    pl = rules.get("price_limits") or rules.get("price_limit") or {}
    if isinstance(pl, dict) and spot in pl:
        v = pl[spot]
        if isinstance(v, dict):
            v = v.get("mean", v.get("typical", v.get("mu")))
        if num(v, 0) > 3:
            p["mu"] = num(v)
    if "game" in ev:
        p["mu"] += 0.6
        mix = [mix[0] + 0.10, mix[1] - 0.05, mix[2] - 0.05]
    if "midterm" in ev and spot == "thompson":
        mix = [0.30, 0.35, 0.35]
    if "concert" in ev and spot == "union":
        p["mu"] += 0.6
    if "late" in ev and spot in ("highst", "northres"):
        mix = [0.65, 0.30, 0.05]
        p["mu"] -= 0.3
    s = sum(max(0, m) for m in mix)
    p["mix"] = [max(0, m) / s for m in mix]
    return p


def rivals_for(state, spot, N):
    last = state.get("last_day") or {}
    offs = []
    for o in offers_in(spot_entries(last).get(spot) or {}):
        if is_me(o, spot, state):
            continue
        q = o.get("quality", "standard")
        if q not in QV:
            continue
        rep = rival_rep(tname(o), spot, state)
        offs.append((QV[q] * rf(rep), num(o.get("price"), 10.0), QRANK[q], num(o.get("meals"), 100)))
    # Crowded spots attract newcomers: assume at least some trucks per customer.
    want = (0.5 if last.get("day") else 0.8) * N / 110.0
    for _ in range(max(0, int(round(want)) - len(offs))):
        offs.append((12.0, 10.5, 1, 120.0))
    return offs


def eval_spot(spot, N, myrep, offs, prm, budget, days_left, seed):
    rng = random.Random(seed)
    mix0, mix1 = prm["mix"][0], prm["mix"][1]
    samples = []
    for _ in range(N_SAMPLES):
        L = rng.gauss(prm["mu"], prm["sd"])
        u = rng.random()
        q = 0 if u < mix0 else (1 if u < mix0 + mix1 else 2)
        best, bp = None, 0.0
        for (val, pr, qr, _m) in offs:
            if pr <= L and qr >= q:
                v = val - pr
                if v >= 0 and (best is None or v > best + 1e-9 or (abs(v - best) < 1e-9 and pr < bp)):
                    best, bp = v, pr
        samples.append((L, q, best, bp))
    rival_meals = sum(o[3] for o in offs)

    top = None
    for qual in QUALS:
        c = COST[qual]
        qr = QRANK[qual]
        myval = QV[qual] * rf(myrep)
        cap = int(budget / c + 1e-9)
        for p in PRICES:
            if p <= c * 1.1:
                continue
            v = myval - p
            if v < 0:
                break
            mine = rchosen = racc = 0
            for (L, q, b, bp) in samples:
                acc = q <= qr and p <= L
                if acc and (b is None or v > b + 1e-9 or (abs(v - b) < 1e-9 and p < bp)):
                    mine += 1
                elif b is not None:
                    rchosen += 1
                    if acc:
                        racc += 1
            overflow = max(0.0, rchosen / N_SAMPLES * N - rival_meals)
            spill = 0.6 * overflow * (racc / rchosen if rchosen else 0)
            D = (mine / N_SAMPLES * N + spill) * BRAIN.calib
            if D < 0.5:
                continue
            cr = clamp(1 - c / p, 0.02, 0.98)
            m = int(round(D * math.exp(SIGMA * NormalDist().inv_cdf(cr) - SIGMA ** 2 / 2)))
            m = clamp(m, 0, cap)
            ds = [D * math.exp(SIGMA * z - SIGMA ** 2 / 2) for z in ZS]
            esold = sum(min(d, m) for d in ds) / len(ds)
            ev = p * esold - c * m
            dr = clamp(myrep + RDELTA[qual], 0, 100) - myrep
            bonus = dr * 0.006 * QV[qual] * 0.5 * esold * min(max(days_left - 1, 0), 6) * 0.5
            score = ev + bonus
            if top is None or score > top[0]:
                top = (score, ev, {"spot": spot, "price": p, "quality": qual, "meals": int(m)}, D, esold)
    return top


def decide(state):
    ingest(state)
    rules = state.get("rules") or {}
    for q, c in (rules.get("costs") or {}).items():
        if q in COST:
            COST[q] = num(c, COST[q])
    for q, v in (rules.get("quality_values") or {}).items():
        if q in QV:
            QV[q] = num(v, QV[q])
    budget = num(rules.get("budget"), 500.0)
    spots = [s for s in (rules.get("spots") or SPOTS) if s in BASE] or list(SPOTS)

    fc = state.get("forecast") or {}
    exp_c = fc.get("expected_customers") or {}
    ntr = max(4, len(state.get("trucks") or []))
    ev = events_text(state)
    myrep_all = (state.get("you") or {}).get("reputation") or {}

    mode = state.get("mode")
    total = num(state.get("days") or state.get("total_days") or rules.get("days"),
                {"final": 30, "live": 48}.get(mode, 30))
    ofd = int(num(state.get("order_for_day"), 1))
    days_left = int(total - ofd + 1)

    best = None
    for s in spots:
        N = num(exp_c.get(s) if isinstance(exp_c, dict) else None, SHARE[s] * 110 * ntr)
        if N <= 0:
            continue
        prm = spot_params(s, ev, rules)
        offs = rivals_for(state, s, N)
        seed = "%s|%s|%s|%s" % (mode, state.get("market"), ofd, s)
        r = eval_spot(s, N, num(myrep_all.get(s), 50.0), offs, prm, budget, days_left, seed)
        if r and (best is None or r[0] > best[0]):
            best = r

    if best is None or best[0] <= 0:
        log("  no profitable option, sitting out")
        return {"spot": "union", "price": 12.0, "quality": "standard", "meals": 0}
    score, evp, order, D, esold = best
    BRAIN.pred[ofd] = D
    log("  plan: %s  demand~%.0f  exp sold~%.0f  exp profit $%.0f  calib %.2f" % (
        order, D, esold, evp, BRAIN.calib))
    return order


def safe_decide(state):
    if DUMP:
        try:
            with open("last_state.json", "w") as f:
                json.dump(state, f, indent=1)
        except Exception:
            pass
    try:
        t = time.time()
        o = decide(state)
        log("  (decided in %.2fs)" % (time.time() - t))
        return o
    except Exception:
        traceback.print_exc()
        return {"spot": "union", "price": 12.5, "quality": "premium", "meals": 60}


# ---------------------------------------------------------------- plumbing (from the starter)

def call(method, path, body=None, timeout=35):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(URL + path, data=data, method=method, headers={
        "Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except Exception:
            return e.code, {"error": str(e)}


def play(base):
    BRAIN.reset(None)
    version, ordered_for, backoff = 0, None, 1
    while True:
        try:
            status, state = call("GET", base + "/state?since=%d" % version)
        except (urllib.error.URLError, OSError) as e:
            print("network error (%s), retrying in %ds" % (e, backoff))
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue
        backoff = 1
        if status == 429:
            time.sleep(2)
            continue
        if status != 200:
            print("error %d: %s" % (status, state.get("error")))
            if status in (401, 404):
                sys.exit(1)
            time.sleep(3)
            continue
        version = state["version"]
        if state["phase"] == "over" and state["mode"] == "practice":
            report(state)
            return state
        key = (state.get("mode"), state.get("market"), state.get("order_for_day"))
        if state["phase"] == "night" and state.get("orders_open") and key != ordered_for:
            report(state)
            order = safe_decide(state)
            status, res = call("POST", base + "/orders", order)
            if status == 200:
                ordered_for = key
                log("  ordered for day %s: %s" % (res.get("for_day"), order))
                if "state" in res:
                    version = 0
            else:
                print("  order rejected (%d): %s" % (status, res.get("error")))
                time.sleep(1)


def report(state):
    if QUIET:
        return
    last = state.get("last_day") or {}
    mine = last.get("you") or {}
    where = state["mode"] + (" market %s" % state["market"] if state["mode"] == "final" else "")
    if mine.get("spot"):
        print("[%s] day %s: %s at %s $%s, sold %s/%s, profit $%.2f, cash $%.2f" % (
            where, last.get("day"), mine.get("quality"), mine.get("spot"), mine.get("price"),
            mine.get("sold"), mine.get("meals"), num(mine.get("profit")), num(state["you"].get("cash"))))
    elif last.get("day"):
        print("[%s] day %s: sat out, cash $%.2f" % (where, last.get("day"), num(state["you"].get("cash"))))
    else:
        print("[%s] night before day %s, cash $%.2f" % (where, state.get("order_for_day"), num(state["you"].get("cash"))))


def practice(seed=None, days=30, rivals=3):
    body = {"days": days, "rivals": rivals}
    if seed:
        body["seed"] = seed
    status, res = call("POST", "/api/v1/practice", body)
    if status not in (200, 201):
        sys.exit("could not start practice (%d): %s" % (status, res.get("error")))
    final = play(res["base"])
    return final


def main():
    global DUMP, QUIET
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--practice", action="store_true")
    ap.add_argument("--bench", type=int, default=0, help="run N practice markets and average")
    ap.add_argument("--seed")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--rivals", type=int, default=3)
    ap.add_argument("--dump", action="store_true")
    args = ap.parse_args()
    DUMP = args.dump
    if not KEY:
        sys.exit("Set NIGHTSHIFT_KEY first.")
    if args.bench:
        QUIET = True
        profits, wins = [], 0
        for i in range(args.bench):
            f = practice("bench-%d" % i, args.days, args.rivals)
            me = num(f["you"].get("cash"))
            ranked = sorted(f["trucks"], key=lambda t: -num(t.get("cash")))
            place = 1 + sum(1 for t in ranked if num(t.get("cash")) > me + 1e-6)
            wins += place == 1
            profits.append(me)
            print("market %2d  profit $%9.2f  place %d/%d" % (i, me, place, len(ranked)))
        print("average profit $%.2f   won %d/%d" % (sum(profits) / len(profits), wins, len(profits)))
    elif args.practice:
        f = practice(args.seed, args.days, args.rivals)
        print("practice over: cash $%.2f" % num(f["you"].get("cash")))
        for t in sorted(f["trucks"], key=lambda t: -num(t.get("cash"))):
            print("  %-24s $%10.2f" % (t.get("truck_name"), num(t.get("cash"))))
    else:
        print("playing the live game at %s (Ctrl+C to stop)" % URL)
        play("/api/v1")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
