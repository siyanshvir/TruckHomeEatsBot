"""Offline simulator of the documented Food Truck Wars rules (no network, no key).

Used only to test bot logic and compare versions on fixed seeds. Its hidden
customer parameters are GUESSES that differ from the bot's priors on purpose;
real practice markets are the ground truth.

    python3 sim_local.py                 # starter vs current bot, fixed seeds
"""
import importlib.util
import json
import math
import random
import statistics
import sys
import time

SPOTS = ["highst", "oval", "union", "northres", "thompson", "stadium"]
SHARE = {"highst": .24, "oval": .20, "union": .18, "northres": .16, "thompson": .14, "stadium": .08}
COSTS = {"cheap": 2.50, "standard": 3.75, "premium": 5.50}
QVAL = {"cheap": 8, "standard": 12, "premium": 16}
QI = {"cheap": 0, "standard": 1, "premium": 2}
STEP = {"cheap": -3, "standard": 2, "premium": 5}
WEATHERS = ["clear", "cloudy", "rain", "cold"]
# hidden truth (differs from bot priors)
LIMIT = {"highst": (12.0, 6.0), "oval": (13.5, 3.0), "union": (11.5, 3.0),
         "northres": (10.5, 2.5), "thompson": (13.0, 4.0), "stadium": (16.0, 4.0)}
MINQ = {"stadium": (.10, .40, .50)}
DMINQ = (.40, .42, .18)
RAIN = {"oval": .55, "stadium": .8, "highst": .9, "union": 1.15, "thompson": 1.1, "northres": 1.0}
COLD = {"oval": .8, "stadium": .85}


def house_bot(kind, rng):
    def f(state):
        ld = state.get("last_day") or {}
        if kind == "starter":
            return {"spot": rng.choice(SPOTS), "price": 10, "quality": "standard", "meals": 133}
        if kind == "cheap":
            return {"spot": rng.choice(["highst", "northres", "union"]), "price": 6, "quality": "cheap", "meals": 200}
        if kind == "premium":
            return {"spot": "highst", "price": 13, "quality": "premium", "meals": 90}
        if kind == "follower":
            best, bp = rng.choice(SPOTS), -1e9
            for s, info in (ld.get("spots") or {}).items():
                for t in info["trucks"]:
                    if t["profit"] > bp:
                        best, bp = s, t["profit"]
            return {"spot": best, "price": 11, "quality": "standard", "meals": 120}
        if kind == "home":
            return {"spot": "oval", "price": 12, "quality": "premium", "meals": 80}
        return {"spot": "union", "price": 9, "quality": "standard", "meals": 133}
    return f


def run_market(decide, seed, nrivals, days=30, timing=None):
    rng = random.Random(seed)
    kinds = ["starter", "cheap", "premium", "follower", "home", "flat"]
    rivals = [("rival%d" % i, house_bot(rng.choice(kinds), random.Random(seed * 31 + i)))
              for i in range(nrivals)]
    names = ["me"] + [n for n, _ in rivals]
    rep = {n: {s: 50.0 for s in SPOTS} for n in names}
    cash = {n: 0.0 for n in names}
    last_day = None
    total_trucks = max(4, len(names))
    stats = {"sold": 0, "meals": 0, "revenue": 0.0, "food": 0.0, "spots": {}, "quals": {}, "prices": []}
    for day in range(1, days + 1):
        wp = [rng.uniform(.2, .5), rng.uniform(.1, .4), rng.uniform(0, .4), rng.uniform(0, .2)]
        tot = sum(wp)
        wp = [x / tot for x in wp]
        events = {}
        if rng.random() < .15:
            events["stadium"] = 4.0     # game day
        if rng.random() < .1:
            events["thompson"] = 1.6
        expected = {s: 110 * total_trucks * SHARE[s] * events.get(s, 1.0) for s in SPOTS}
        state = {
            "mode": "practice", "market": seed, "phase": "night", "order_for_day": day, "days": days,
            "rules": {"budget": 500, "costs": COSTS, "quality_values": QVAL, "spots": SPOTS},
            "forecast": {"weather": dict(zip(WEATHERS, wp)), "expected_customers": dict(expected),
                         "events": list(events)},
            "you": {"truck_name": "me", "cash": cash["me"], "reputation": dict(rep["me"])},
            "trucks": [{"truck_name": n, "cash": cash[n],
                        "reputation": sum(rep[n].values()) / 6} for n in names],
            "last_day": last_day,
        }
        t0 = time.time()
        orders = {"me": decide(json.loads(json.dumps(state)))}
        if timing is not None:
            timing.append(time.time() - t0)
        for n, f in rivals:
            orders[n] = f(state)
        w = rng.choices(WEATHERS, wp)[0]
        spots_out = {}
        for s in SPOTS:
            mult = (RAIN.get(s, 1) if w == "rain" else COLD.get(s, 1) if w == "cold" else 1.0)
            n_c = max(0, int(rng.gauss(expected[s] * mult, expected[s] * .12)))
            here = []
            for n in names:
                o = orders[n]
                if o.get("spot") != s or o.get("meals", 0) <= 0:
                    continue
                q = o["quality"]
                meals = int(min(o["meals"], 500 // COSTS[q]))
                here.append({"name": n, "price": float(o["price"]), "quality": q, "meals": meals,
                             "stock": meals, "sold": 0,
                             "value": QVAL[q] * (0.7 + 0.6 * rep[n][s] / 100) - float(o["price"])})
            lm, ls = LIMIT[s]
            mq = MINQ.get(s, DMINQ)
            hungry = 0
            for _ in range(n_c):
                lim = rng.gauss(lm, ls)
                u = rng.random()
                q = 0 if u < mq[0] else 1 if u < mq[0] + mq[1] else 2
                best = None
                for t in here:
                    if t["stock"] <= 0 or t["price"] > lim or QI[t["quality"]] < q or t["value"] < 0:
                        continue
                    if best is None or t["value"] > best["value"] or (
                            t["value"] == best["value"] and t["price"] < best["price"]):
                        best = t
                if best:
                    best["stock"] -= 1
                    best["sold"] += 1
                else:
                    hungry += 1
            trucks = []
            for t in here:
                prof = t["sold"] * t["price"] - t["meals"] * COSTS[t["quality"]]
                cash[t["name"]] += prof
                if t["sold"] > 0:
                    rep[t["name"]][s] = max(0, min(100, rep[t["name"]][s] + STEP[t["quality"]]))
                trucks.append({"truck_name": t["name"], "price": t["price"], "quality": t["quality"],
                               "meals": t["meals"], "sold": t["sold"], "profit": prof})
                if t["name"] == "me":
                    stats["sold"] += t["sold"]
                    stats["meals"] += t["meals"]
                    stats["revenue"] += t["sold"] * t["price"]
                    stats["food"] += t["meals"] * COSTS[t["quality"]]
                    stats["spots"][s] = stats["spots"].get(s, 0) + 1
                    stats["quals"][t["quality"]] = stats["quals"].get(t["quality"], 0) + 1
                    stats["prices"].append(t["price"])
            spots_out[s] = {"customers": n_c, "hungry": hungry, "trucks": trucks}
        for n in names:   # drift unvisited spots toward 50
            for s in SPOTS:
                if orders[n].get("spot") != s:
                    rep[n][s] += (50 - rep[n][s]) * 0.03
        mine = next((t for info in spots_out.values() for t in info["trucks"] if t["truck_name"] == "me"), {})
        last_day = {"day": day, "weather": w, "spots": spots_out, "you": dict(mine, spot=orders["me"].get("spot"))}
    stats["waste"] = stats["meals"] - stats["sold"]
    stats["avg_rep"] = sum(rep["me"].values()) / 6
    return cash["me"], stats


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


if __name__ == "__main__":
    seeds = [1, 2, 3, 4, 5]
    counts = [3, 5, 8]
    if "--holdout" in sys.argv:
        seeds = [101, 102, 103, 104, 105]
    paths = [("starter", "backup/bot_starter.py"), ("current", "bot.py")]
    results = {}
    for label, path in paths:
        m = load(path, "bot_" + label)
        rows, timing = [], []
        for c in counts:
            for sd in seeds:
                random.seed(sd)                       # starter uses global random
                prof, st = run_market(m.decide, sd * 1000 + c, c, timing=timing)
                rows.append({"seed": sd, "rivals": c, "profit": round(prof, 2),
                             "waste": st["waste"], "avg_rep": round(st["avg_rep"], 1),
                             "spots": st["spots"], "quals": st["quals"],
                             "avg_price": round(statistics.mean(st["prices"]), 2) if st["prices"] else None})
        results[label] = rows
        p = [r["profit"] for r in rows]
        print("%-8s mean $%8.0f  median $%8.0f  sd $%6.0f  worst $%8.0f  max decide %.3fs" % (
            label, statistics.mean(p), statistics.median(p), statistics.pstdev(p), min(p), max(timing)))
        for c in counts:
            pc = [r["profit"] for r in rows if r["rivals"] == c]
            print("         rivals=%d  mean $%8.0f" % (c, statistics.mean(pc)))
    with open("sim_results.json", "w") as f:
        json.dump(results, f, indent=1)
