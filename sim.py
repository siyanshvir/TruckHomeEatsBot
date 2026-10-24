"""Offline simulator for Night Shift, used to tune strategy.py without calling the server.

It implements the published rules (customer choice, sell-outs, reputation, drift, budget) with
hidden per-market customer distributions randomized around the published descriptions, plus
a few house-bot styles and optional "smart" rivals that run our own Brain.

    python3 sim.py                      # compare strategies on the default seed set
    python3 sim.py --markets 40 --rivals 3 --smart 1
"""
import argparse
import math
import random
import statistics
import sys
import time

import strategy as S

WEATHERS = ("clear", "cloudy", "rain", "cold")
EVENT_KINDS = ("game", "midterm", "concert", "late", "rally")
EVENT_LABEL = {"game": "Game day", "midterm": "Midterms", "concert": "Concert at the Union",
               "late": "Late-night rush", "rally": "Rally on the Oval"}
EVENT_KEY = {"game": "gameday", "midterm": "midterms", "concert": "concert", "late": "latenight", "rally": "rally"}


def event_list(ev):
    """Events in the server's format."""
    return [{"key": EVENT_KEY[k], "name": EVENT_LABEL[k]} for k in ev]


class Market:
    def __init__(self, seed, days=30):
        self.rng = random.Random(seed)
        self.days = days
        rng = self.rng
        # hidden truth: per-spot price limit distribution and quality-minimum mix
        self.truth = {}
        for s, (mu, sd, qmix, share) in S.PRIORS.items():
            q = S.shift_qmix(qmix, rng.uniform(-0.08, 0.08))
            self.truth[s] = (mu + rng.uniform(-1.2, 1.2), sd * rng.uniform(0.8, 1.25), q, share)
        self.wmult = {w: {s: S.weather_mult(w, s) * rng.uniform(0.9, 1.1) for s in S.SPOTS} for w in WEATHERS}
        self.schedule = []
        for d in range(days + 2):
            ev = [rng.choice(EVENT_KINDS)] if rng.random() < 0.40 else []
            if ev and rng.random() < 0.15:
                ev = sorted(set(ev + [rng.choice(EVENT_KINDS)]))
            raw = {"clear": rng.random() * 2 + 0.5, "cloudy": rng.random(), "rain": rng.random() * 1.2,
                   "cold": rng.random() * 0.8}
            tot = sum(raw.values())
            probs = {k: round(v / tot, 2) for k, v in raw.items()}
            r, acc, w = rng.random(), 0.0, "clear"
            for k, p in probs.items():
                acc += p
                if r <= acc:
                    w = k
                    break
            self.schedule.append((probs, ev, w))

    def expected(self, day, n_trucks):
        probs, ev, _ = self.schedule[day]
        out = {}
        for s in S.SPOTS:
            base = 110.0 * max(4, n_trucks) * self.truth[s][3]
            for k in ev:
                cr = S.EVENTS[k]["crowd"]
                base *= cr.get(s, cr.get("*", 1.0))
            out[s] = round(base * sum(p * self.wmult[w][s] for w, p in probs.items()))
        return out

    def spot_truth(self, s, ev):
        mu, sd, qmix, _ = self.truth[s]
        for k in ev:
            e = S.EVENTS[k]
            mu += e.get("mu", {}).get(s, e.get("mu", {}).get("*", 0.0))
            if s in e.get("qmix", {}):
                qmix = e["qmix"][s]
        return mu, sd, qmix


def play_day(market, day, orders, reps):
    """orders: {name: order or None}. Returns per-spot results."""
    rng = market.rng
    probs, ev, weather = market.schedule[day]
    n_trucks = len(orders)
    results = {}
    for s in S.SPOTS:
        base = 110.0 * max(4, n_trucks) * market.truth[s][3]
        for k in ev:
            cr = S.EVENTS[k]["crowd"]
            base *= cr.get(s, cr.get("*", 1.0))
        n = max(0, int(round(rng.gauss(base * market.wmult[weather][s], 0.08 * base))))
        here = [(name, o) for name, o in orders.items() if o and o["spot"] == s]
        mu, sd, qmix = market.spot_truth(s, ev)
        left = {name: int(o["meals"]) for name, o in here}
        sold = {name: 0 for name, _ in here}
        hungry = 0
        for _ in range(n):
            limit = rng.gauss(mu, sd)
            r = rng.random()
            minq = 0 if r < qmix[0] else (1 if r < qmix[0] + qmix[1] else 2)
            best, bv = None, None
            for name, o in here:
                qi = S.qindex(o["quality"])
                if left[name] <= 0 or o["price"] > limit or qi < minq:
                    continue
                v = S.VALUE[qi] * S.rep_factor(reps[name][s]) - o["price"]
                if v < 0:
                    continue
                if best is None or v > bv or (v == bv and o["price"] < orders[best]["price"]):
                    best, bv = name, v
            if best is None:
                hungry += 1
            else:
                left[best] -= 1
                sold[best] += 1
        results[s] = {"customers": n, "hungry": hungry,
                      "trucks": [{"id": name, "truck_name": name, "price": o["price"], "quality": o["quality"],
                                  "meals": int(o["meals"]), "sold": sold[name]} for name, o in here]}
    # reputation
    for name, o in orders.items():
        for s in S.SPOTS:
            if o and o["spot"] == s:
                if sold_at(results, s, name) >= 1:
                    reps[name][s] = max(0.0, min(100.0, reps[name][s] + S.REP_STEP[S.qindex(o["quality"])]))
            else:
                reps[name][s] += 0.03 * (50.0 - reps[name][s])
    return results, weather, ev


def sold_at(results, s, name):
    for t in results[s]["trucks"]:
        if t["truck_name"] == name:
            return t["sold"]
    return 0


# ---------------------------------------------------------------- house bots

class House:
    def __init__(self, kind, rng):
        self.kind, self.rng = kind, rng
        self.home = rng.choice(S.SPOTS)
        self.price = round(rng.uniform(9.0, 12.0), 2)

    def decide(self, state):
        rng = self.rng
        if self.kind == "random":       # the starter bot
            return {"spot": rng.choice(S.SPOTS), "price": 10.0, "quality": "standard", "meals": 133}
        if self.kind == "loyal":
            return {"spot": self.home, "price": self.price, "quality": "standard", "meals": 120}
        if self.kind == "cheap":
            return {"spot": rng.choice(("highst", "thompson", "northres")), "price": 6.5, "quality": "cheap",
                    "meals": 160}
        if self.kind == "premium":
            return {"spot": rng.choice(("union", "oval")), "price": 14.0, "quality": "premium", "meals": 80}
        if self.kind == "chaser":       # goes where the most customers are expected
            exp = state["forecast"]["expected_customers"]
            return {"spot": max(exp, key=exp.get), "price": 11.0, "quality": "standard", "meals": 133}
        events = state["forecast"]["events"]
        keys = [e["key"] for e in events]
        if self.kind == "eddie":         # the practice server's house bots
            return {"spot": "union", "price": 10.5, "quality": "standard", "meals": 110}
        if self.kind == "fran":
            return {"spot": "thompson", "price": 14.5, "quality": "premium", "meals": 60}
        if self.kind == "dan":
            spot = {"rally": "oval", "midterms": "thompson", "concert": "union"}.get(keys[0] if keys else "", "highst")
            return {"spot": spot, "price": 6.5, "quality": "cheap", "meals": 180}
        raise ValueError(self.kind)


class Smart:
    """A rival running our own engine (stand-in for clever live opponents)."""
    def __init__(self, params, seed):
        self.brain = S.Brain(params, seed=seed)

    def decide(self, state):
        return self.brain.decide(state)


class Starter:
    def decide(self, state):
        return {"spot": random.choice(S.SPOTS), "price": 10.0, "quality": "standard", "meals": 133}


def run_market(seed, players, rivals=3, smart=0, days=30, rival_kinds=None):
    """players: {name: bot}. Rivals are house bots (and `smart` copies of our Brain). Returns profits."""
    market = Market(seed, days)
    rng = random.Random(seed * 7 + 1)
    kinds = rival_kinds or ("eddie", "fran", "dan", "random", "loyal", "cheap", "premium", "chaser")
    bots = dict(players)
    for i in range(rivals):
        kind = kinds[i] if rival_kinds is None and i < 3 else rng.choice(kinds)
        bots["house%d" % i] = House(kind, random.Random(seed * 31 + i))
    for i in range(smart):
        bots["smart%d" % i] = Smart(None, seed=seed * 13 + i)
    reps = {name: {s: 50.0 for s in S.SPOTS} for name in bots}
    cash = {name: 0.0 for name in bots}
    last_day = None
    for day in range(1, days + 1):
        probs, ev, _ = market.schedule[day]
        orders = {}
        for name, bot in bots.items():
            state = {
                "phase": "night", "orders_open": True, "order_for_day": day, "mode": "practice",
                "market": seed, "day": day - 1, "days": days,
                "forecast": {"weather": probs, "events": event_list(ev),
                             "expected_customers": market.expected(day, len(bots))},
                "you": {"id": name, "truck_name": name, "cash": cash[name], "reputation": dict(reps[name])},
                "last_day": last_day and dict(last_day, you=last_day["by"].get(name, {})),
                "trucks": [{"id": n, "truck_name": n, "cash": cash[n], "reputation": dict(reps[n])} for n in bots],
            }
            try:
                o = bot.decide(state)
            except Exception as e:      # a crashing bot sits out, like on the server
                print("bot %s crashed: %r" % (name, e), file=sys.stderr)
                o = None
            if o:
                o = dict(o)
                qi = S.qindex(o["quality"])
                o["meals"] = max(0, min(int(o["meals"]), int(S.BUDGET // S.COST[qi])))
                o["price"] = max(1.0, min(30.0, float(o["price"])))
            orders[name] = o
        results, weather, ev = play_day(market, day, orders, reps)
        by = {}
        for name, o in orders.items():
            if not o:
                by[name] = {}
                continue
            sold = sold_at(results, o["spot"], name)
            profit = sold * o["price"] - o["meals"] * S.COST[S.qindex(o["quality"])]
            cash[name] += profit
            by[name] = dict(o, sold=sold, profit=profit)
        last_day = {"day": day, "weather": weather, "events": event_list(ev),
                    "spots": results, "by": by}
    return cash


SEEDS = list(range(1001, 1021))


def evaluate(make_bot, seeds=SEEDS, rivals=3, smart=0, days=30):
    profits = []
    for seed in seeds:
        cash = run_market(seed, {"me": make_bot(seed)}, rivals=rivals, smart=smart, days=days)
        profits.append(cash["me"])
    return statistics.mean(profits), profits


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--markets", type=int, default=20)
    ap.add_argument("--rivals", type=int, default=3)
    ap.add_argument("--smart", type=int, default=0)
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()
    seeds = list(range(1001, 1001 + args.markets))
    t = time.time()
    base, _ = evaluate(lambda s: Starter(), seeds, args.rivals, args.smart, args.days)
    new, prof = evaluate(lambda s: S.Brain(seed=s), seeds, args.rivals, args.smart, args.days)
    print("starter bot: avg profit $%.0f" % base)
    print("strategy:    avg profit $%.0f  (min $%.0f, max $%.0f)  [%.1fs]" % (
        new, min(prof), max(prof), time.time() - t))


if __name__ == "__main__":
    main()
