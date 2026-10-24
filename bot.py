"""Food Truck Wars starter bot. Python 3.8+, standard library only.

    export NIGHTSHIFT_KEY=fk_your_key_here
    python3 bot.py              # play the live game (leave it running)
    python3 bot.py --practice   # play a private practice market and print your profit

It works, but it plays badly. Rewrite decide() below.
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from statistics import NormalDist

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


# ---------------------------------------------------------------- customer model
# Fitted from practice markets with tools/fit.py. Per spot: a customer's price limit is
# Normal(mu, sd), and pc / pp are the shares whose minimum quality is cheap / premium
# (everyone else needs at least standard).
POP = {
    "oval": (13.05, 2.98, 0.35, 0.20),
    "thompson": (12.09, 2.78, 0.36, 0.20),
    "union": (12.71, 3.27, 0.33, 0.18),
    "stadium": (13.05, 3.07, 0.40, 0.28),
    "highst": (12.05, 2.96, 0.51, 0.20),
    "northres": (10.91, 2.72, 0.52, 0.07),
}
QV = {"cheap": 8.0, "standard": 12.0, "premium": 16.0}
QRANK = {"cheap": 0, "standard": 1, "premium": 2}
REP_GAIN = {"cheap": -3, "standard": 2, "premium": 5}
K = 16                                    # price-limit quantiles per spot
_Z = [NormalDist().inv_cdf((i + 0.5) / K) for i in range(K)]
_CELLS = {}


def population(mu, sd, pc, pp):
    """(price limit, minimum quality rank, share of a spot's customers) on a fixed grid."""
    mins = [(0, pc), (1, max(0.0, 1 - pc - pp)), (2, pp)]
    return [(mu + sd * z, r, p / K) for z in _Z for r, p in mins if p > 0]


def cells_for(spot):
    if spot not in _CELLS:
        _CELLS[spot] = population(*POP.get(spot, (12.0, 2.9, 0.35, 0.2)))
    return _CELLS[spot]


def rep_factor(rep):
    return 0.7 + 0.6 * rep / 100.0


def expected_sold(count, cells, offers):
    """Expected meals sold by each offer (dict: price, quality, meals, rep) among `count` customers.
    Customers skip sold-out trucks, prices above their limit, quality below their minimum and
    anything worth nothing; the rest buy the best value, ties to the cheaper truck."""
    n = len(offers)
    vals = [QV[o["quality"]] * rep_factor(o["rep"]) - o["price"] for o in offers]
    rank = [QRANK[o["quality"]] for o in offers]
    price = [o["price"] for o in offers]
    sold = [0.0] * n
    out = [False] * n
    w = [c[2] * count for c in cells]
    for _ in range(n + 1):
        demand = [0.0] * n
        pick = []
        for ci, (lim, mr, _) in enumerate(cells):
            best = -1
            if w[ci] > 0:
                for i in range(n):
                    if out[i] or vals[i] <= 0 or price[i] > lim or rank[i] < mr:
                        continue
                    if best < 0 or vals[i] > vals[best] + 1e-9 or (abs(vals[i] - vals[best]) <= 1e-9 and price[i] < price[best]):
                        best = i
            pick.append(best)
            if best >= 0:
                demand[best] += w[ci]
        over = [i for i in range(n) if not out[i] and demand[i] > offers[i]["meals"] - sold[i] + 1e-9]
        if not over:
            for i in range(n):
                sold[i] += demand[i]
            break
        frac = {}
        for i in over:
            room = offers[i]["meals"] - sold[i]
            frac[i] = room / demand[i]
            sold[i] += room
            out[i] = True
        nxt = []
        for ci, b in enumerate(pick):
            if b in frac:
                nxt.append(w[ci] * (1 - frac[b]))      # not served: look for the next best truck
            else:
                if b >= 0:
                    sold[b] += w[ci]
                nxt.append(0.0)
        w = nxt
    return sold


# ---------------------------------------------------------------- strategy

GENERIC = {"price": 10.5, "quality": "standard", "meals": 110, "rep": 50}   # an unknown newcomer
PRICE_RANGE = {"cheap": (4.0, 10.5), "standard": (6.5, 13.5), "premium": (9.5, 17.0)}
NOISE = ((0.25, 0.85), (0.5, 1.0), (0.25, 1.15))    # day-to-day swing in customers
HORIZON = 6          # how many future days a reputation point is worth counting
REP_WEIGHT = 0.6     # discount on that future value


def best_meals(demands, price, cost, cap):
    """Meals that maximise average profit over [(weight, demand)] scenarios."""
    best_m, best_p = 0, 0.0
    for m in sorted({0} | {min(cap, int(d + 0.999)) for _, d in demands}):
        p = sum(w * min(m, d) * price for w, d in demands) - m * cost
        if p > best_p:
            best_m, best_p = m, p
    return best_m, best_p


def scenarios(spot, rivals_here, have_history, rel_crowd):
    """Guesses at who else parks at `spot` tomorrow, with weights."""
    if not have_history:
        return [(0.5, []), (0.5, [GENERIC])]
    extra = 0.1 + 0.3 * rel_crowd                   # busy spots attract newcomers
    return [(0.5, rivals_here), (0.5 - extra, []), (extra, rivals_here + [GENERIC])]


def demand_for(count, cells, scen, mine):
    """[(weight, expected demand)] for my offer `mine` (unlimited meals) across scenarios and noise."""
    out = []
    for ws, rivals in scen:
        d = expected_sold(count, cells, [dict(mine, meals=10 ** 6)] + rivals)[0]
        for wn, mult in NOISE:
            out.append((ws * wn, d * mult))
    return out


def plan(state):
    rules = state["rules"]
    costs, budget = rules["costs"], rules["budget"]
    me = state["you"]["id"]
    expect = state["forecast"]["expected_customers"]
    crowd_max = max(expect.values()) or 1
    last = state.get("last_day") or {}
    reps = {t["id"]: t.get("reputation") or {} for t in state["trucks"]}
    days_left = max(1, state.get("days", 30) - state.get("day", 0))
    horizon = min(days_left - 1, HORIZON) * REP_WEIGHT

    rivals = {s: [] for s in rules["spots"]}
    for s, info in (last.get("spots") or {}).items():
        for t in info["trucks"]:
            if t["id"] != me and t.get("meals"):
                rivals[s].append({"price": t["price"], "quality": t["quality"], "meals": t["meals"],
                                  "rep": reps.get(t["id"], {}).get(s, 50)})

    best = None
    for spot in rules["spots"]:
        count = expect[spot]
        cells = cells_for(spot)
        scen = scenarios(spot, rivals[spot], bool(last.get("spots")), expect[spot] / crowd_max)
        my_rep = state["you"]["reputation"][spot]
        for q in ("cheap", "standard", "premium"):
            cap = int(budget / costs[q] + 1e-9)
            lo, hi = PRICE_RANGE[q]
            p = lo
            while p <= hi + 1e-9:
                mine = {"price": p, "quality": q, "rep": my_rep}
                m, profit = best_meals(demand_for(count, cells, scen, mine), p, costs[q], cap)
                score = profit
                if horizon and m and my_rep < 100:
                    up = min(100, my_rep + 10)
                    _, profit_up = best_meals(demand_for(count, cells, scen, dict(mine, rep=up)), p, costs[q], cap)
                    score += horizon * REP_GAIN[q] * (profit_up - profit) / (up - my_rep)
                if best is None or score > best[0]:
                    best = (score, spot, q, p, m, profit)
                p += 0.5
    score, spot, q, p, m, profit = best
    if m <= 0 or profit <= 0 and score <= 0:
        return {"spot": spot, "price": round(p, 2), "quality": q, "meals": 0}
    return {"spot": spot, "price": round(p, 2), "quality": q, "meals": int(m)}


def fallback(state):
    """Safe, boring order if the planner ever breaks: best-looking spot, standard, modest meals."""
    expect = state["forecast"]["expected_customers"]
    spot = max(expect, key=expect.get)
    return {"spot": spot, "price": 10.5, "quality": "standard", "meals": 60}


def decide(state):
    """Return tomorrow's order: where to park, what to charge, quality and meals to cook."""
    try:
        return plan(state)
    except Exception as e:
        print("  planner error (%r), using fallback" % (e,))
        return fallback(state)


# ---------------------------------------------------------------- plumbing

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
    """Long-poll the state and send one order per night. Works for live, final and practice."""
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
            order = decide(state)
            status, res = call("POST", base + "/orders", order)
            if status == 200:
                ordered_for = key
                print("  ordered for day %s: %s" % (res.get("for_day"), order))
                if "state" in res:          # practice plays the day right away
                    version = 0
            else:
                print("  order rejected (%d): %s" % (status, res.get("error")))
                time.sleep(1)


def report(state):
    last = state.get("last_day") or {}
    mine = last.get("you") or {}
    where = state["mode"] + (" market %s" % state["market"] if state["mode"] == "final" else "")
    if mine.get("spot"):
        print("[%s] day %s: %s at %s, sold %s/%s, profit $%.2f, cash $%.2f" % (
            where, last.get("day"), mine.get("quality"), mine.get("spot"), mine.get("sold"),
            mine.get("meals"), mine.get("profit", 0), state["you"]["cash"]))
    elif last.get("day"):
        print("[%s] day %s: sat out, cash $%.2f" % (where, last.get("day"), state["you"]["cash"]))
    else:
        print("[%s] night before day %s, cash $%.2f" % (where, state.get("order_for_day"), state["you"]["cash"]))


def main():
    sys.stdout.reconfigure(line_buffering=True)   # show each line right away, even when piped
    ap = argparse.ArgumentParser(description="Food Truck Wars starter bot")
    ap.add_argument("--practice", action="store_true", help="play a private practice market")
    ap.add_argument("--seed", help="practice seed, for repeatable markets")
    ap.add_argument("--days", type=int, default=30, help="practice market length")
    args = ap.parse_args()
    if not KEY:
        sys.exit("Set NIGHTSHIFT_KEY first (you get it when you register).")
    if args.practice:
        body = {"days": args.days}
        if args.seed:
            body["seed"] = args.seed
        status, res = call("POST", "/api/v1/practice", body)
        if status not in (200, 201):
            sys.exit("could not start practice (%d): %s" % (status, res.get("error")))
        print("practice market %s started" % res["id"])
        final = play(res["base"])
        you = final["you"]
        print("practice over: cash $%.2f" % you["cash"])
        for t in sorted(final["trucks"], key=lambda t: -t["cash"]):
            print("  %-24s $%10.2f" % (t["truck_name"], t["cash"]))
    else:
        print("playing the live game at %s (Ctrl+C to stop)" % URL)
        play("/api/v1")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
