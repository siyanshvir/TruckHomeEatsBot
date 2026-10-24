"""Offline simulator built from the official rules (customer model, rep, budget).

Hidden parameters (limit distributions, weather multipliers, event odds, house-bot
behaviour) are NOT published, so this is a stress-test, not ground truth.

    python3 tools/sim.py            # compare strategies across seeds / rival mixes
"""
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bot_v2 as S

SPOTS = S.SPOTS
TRUTH_POP = None   # hidden customer params used by the simulator (default: bot_v2.POP)
WEATHER = ["clear", "cloudy", "rain", "cold"]
WMULT = {  # crowd multiplier by weather
    "clear": dict(highst=1, oval=1, union=1, northres=1, thompson=1, stadium=1),
    "cloudy": dict(highst=.95, oval=.92, union=1.02, northres=.97, thompson=1.03, stadium=.92),
    "rain": dict(highst=.75, oval=.5, union=1.2, northres=.88, thompson=1.2, stadium=.45),
    "cold": dict(highst=.8, oval=.7, union=1.12, northres=.9, thompson=1.12, stadium=.7),
}
EVENTS = [
    ("Game day", "stadium", "game"), ("Midterms", "thompson", "midterm"),
    ("Concert at the Union", "union", "concert"), ("Late-night rush", "highst", "late"),
    ("Rally on the Oval", "oval", "rally"),
]


def event_mult(kind, spot):
    if kind == "game":
        return 5.0 if spot == "stadium" else (1.35 if spot == "highst" else 1.15)
    if kind == "midterm":
        return 1.7 if spot == "thompson" else 1.0
    if kind == "concert":
        return 1.8 if spot == "union" else 1.0
    if kind == "late":
        return 1.8 if spot in ("highst", "northres") else 0.5
    if kind == "rally":
        return 2.0 if spot == "oval" else 1.0
    return 1.0


class Market:
    def __init__(self, seed, bots, days=30, truth=None):
        self.rng = random.Random(seed)
        self.days = days
        self.bots = bots  # list of (name, fn)
        self.n = len(bots)
        self.cash = [0.0] * self.n
        self.rep = [{s: 50.0 for s in SPOTS} for _ in range(self.n)]
        self.truth = truth or {}
        self.weather = "clear"
        self.last_day = None
        self.stats = [dict(rev=0, cost=0, sold=0, meals=0, spots={}, q={}, price=0, nd=0) for _ in range(self.n)]
        # per-market jitter of hidden params
        self.pop = {}
        for s, (mu, sd, pc, pp) in (TRUTH_POP or S.POP).items():
            j = self.truth.get("jitter", 1.0)
            self.pop[s] = (mu + self.rng.gauss(0, 0.8 * j), max(1.0, sd * (1 + self.rng.gauss(0, 0.2 * j))),
                           min(.8, max(0, pc + self.rng.gauss(0, .07 * j))), min(.5, max(0, pp + self.rng.gauss(0, .05 * j))))
        self.next_fc()

    def next_fc(self):
        r = self.rng
        stay = {"rain": .55, "cold": .6}.get(self.weather, .4)
        probs = {w: (1 - stay) / 3 for w in WEATHER}
        probs[self.weather] += stay
        ev = []
        if r.random() < 0.35:
            name, spot, kind = r.choice(EVENTS)
            ev = [{"spot": spot, "name": name, "effect": "", "_k": kind}]
        tot = 110 * max(4, self.n)
        expw = {}
        for s in SPOTS:
            m = sum(probs[w] * WMULT[w][s] for w in WEATHER)
            for e in ev:
                m *= event_mult(e["_k"], s)
            expw[s] = tot * S.SHARE[s] * m
        self.fc = {"weather": probs, "events": [{k: v for k, v in e.items() if k != "_k"} for e in ev],
                   "expected_customers": {s: round(v) for s, v in expw.items()}, "_ev": ev}

    def state(self, i, day):
        return {"mode": "practice", "market": 1, "days": self.days, "order_for_day": day, "phase": "night",
                "rules": {"budget": 500, "costs": dict(S.COST), "values": dict(S.QV), "spots": list(SPOTS),
                          "price_min": 1, "price_max": 30},
                "forecast": {k: v for k, v in self.fc.items() if k != "_ev"},
                "you": {"id": i, "cash": self.cash[i], "reputation": dict(self.rep[i])},
                "last_day": self._ld_for(i), "trucks": [{"id": j} for j in range(self.n)]}

    def _ld_for(self, i):
        if not self.last_day:
            return None
        ld = dict(self.last_day)
        ld["you"] = self.last_day["_you"][i]
        ld.pop("_you", None)
        return ld

    def play_day(self, day):
        r = self.rng
        orders = []
        for i, (name, fn) in enumerate(self.bots):
            try:
                o = fn(self.state(i, day), i, self)
            except Exception as e:
                raise
            q = o["quality"]
            m = min(int(o["meals"]), int(500 // S.COST[q]))
            orders.append(dict(spot=o["spot"], price=float(o["price"]), quality=q, meals=m))
        probs = self.fc["weather"]
        x, w = r.random(), "clear"
        for k in WEATHER:
            x -= probs[k]
            if x <= 0:
                w = k
                break
        self.weather = w
        ev = self.fc["_ev"]
        tot = 110 * max(4, self.n)
        spots_out, you = {}, [None] * self.n
        sold = [0] * self.n
        for s in SPOTS:
            m = WMULT[w][s]
            kinds = [e["_k"] for e in ev]
            for k in kinds:
                m *= event_mult(k, s)
            count = max(0, int(round(tot * S.SHARE[s] * m * r.lognormvariate(0, 0.12))))
            mu, sd, pc, pp = self.pop[s]
            for k in kinds:
                if k in S.EVENT_POP and (k in ("game", "late") or S.EVENTS_SPOT.get(k) == s if hasattr(S, "EVENTS_SPOT") else True):
                    pass
            here = [i for i in range(self.n) if orders[i]["spot"] == s]
            left = {i: orders[i]["meals"] for i in here}
            vals = {i: S.QV[orders[i]["quality"]] * S.rf(self.rep[i][s]) - orders[i]["price"] for i in here}
            hungry = 0
            for _ in range(count):
                lim = r.gauss(mu, sd)
                u = r.random()
                mn = 0 if u < pc else (2 if u > 1 - pp else 1)
                bi = None
                for i in here:
                    o = orders[i]
                    if left[i] <= 0 or o["price"] > lim or S.QR[o["quality"]] < mn or vals[i] < 0:
                        continue
                    if bi is None or vals[i] > vals[bi] or (vals[i] == vals[bi] and o["price"] < orders[bi]["price"]):
                        bi = i
                if bi is None:
                    hungry += 1
                else:
                    left[bi] -= 1
                    sold[bi] += 1
            spots_out[s] = {"customers": count, "hungry": hungry, "trucks": [
                dict(id=i, truck_name=self.bots[i][0], price=orders[i]["price"], quality=orders[i]["quality"],
                     meals=orders[i]["meals"], sold=sold[i]) for i in here]}
        for i in range(self.n):
            o = orders[i]
            rev = sold[i] * o["price"]
            cost = o["meals"] * S.COST[o["quality"]]
            self.cash[i] += rev - cost
            st = self.stats[i]
            st["rev"] += rev; st["cost"] += cost; st["sold"] += sold[i]; st["meals"] += o["meals"]
            st["spots"][o["spot"]] = st["spots"].get(o["spot"], 0) + 1
            st["q"][o["quality"]] = st["q"].get(o["quality"], 0) + 1
            st["price"] += o["price"]; st["nd"] += 1
            for s in SPOTS:
                if s == o["spot"] and sold[i] > 0:
                    self.rep[i][s] = min(100, max(0, self.rep[i][s] + S.REP_DELTA[o["quality"]]))
                elif s != o["spot"]:
                    self.rep[i][s] += (50 - self.rep[i][s]) * 0.03
            you[i] = dict(o, sold=sold[i], revenue=rev, cost=cost, profit=rev - cost)
        self.last_day = {"day": day, "weather": w, "events": self.fc["events"], "spots": spots_out, "_you": you}
        self.next_fc()

    def run(self):
        for d in range(1, self.days + 1):
            self.play_day(d)
        return self.cash


# ------------------------------------------------------------ rival bots

def mk_fixed(spot, price, q, meals):
    def f(st, i, mk):
        return dict(spot=spot, price=price, quality=q, meals=meals)
    return f


def mk_random(price, q):
    def f(st, i, mk):
        return dict(spot=mk.rng.choice(SPOTS), price=price, quality=q, meals=int(500 // S.COST[q]))
    return f


def mk_follower(price, q):
    def f(st, i, mk):
        ld = st.get("last_day")
        if not ld:
            return dict(spot="highst", price=price, quality=q, meals=int(500 // S.COST[q]))
        best = max(SPOTS, key=lambda s: sum(t["sold"] * t["price"] for t in ld["spots"][s]["trucks"]) /
                   max(1, len(ld["spots"][s]["trucks"])))
        return dict(spot=best, price=price, quality=q, meals=int(500 // S.COST[q]))
    return f


def mk_forecast_greedy(q, margin):
    def f(st, i, mk):
        exp = st["forecast"]["expected_customers"]
        s = max(SPOTS, key=lambda x: exp[x])
        return dict(spot=s, price=S.COST[q] + margin, quality=q, meals=int(500 // S.COST[q]))
    return f


def mk_undercut():
    """Adaptive: goes to where the strongest truck was yesterday and beats its value by $1."""
    def f(st, i, mk):
        ld = st.get("last_day")
        if not ld:
            return dict(spot="union", price=10, quality="standard", meals=133)
        bestt, bs = None, None
        for s in SPOTS:
            for t in ld["spots"][s]["trucks"]:
                if t["id"] != i and (bestt is None or t["sold"] * t["price"] > bestt["sold"] * bestt["price"]):
                    bestt, bs = t, s
        if bestt is None:
            return dict(spot="union", price=10, quality="standard", meals=133)
        q = bestt["quality"]
        p = max(S.COST[q] + 1, bestt["price"] - 1)
        return dict(spot=bs, price=p, quality=q, meals=int(500 // S.COST[q]))
    return f


def starter(st, i, mk):
    return dict(spot=mk.rng.choice(SPOTS), price=10.0, quality="standard", meals=133)


def rival_pool(rng, k):
    pool = [
        lambda: ("cheap", mk_fixed(rng.choice(SPOTS), round(rng.uniform(6, 8), 1), "cheap", 200)),
        lambda: ("premium", mk_fixed(rng.choice(SPOTS), round(rng.uniform(11, 14), 1), "premium", 90)),
        lambda: ("std", mk_fixed(rng.choice(SPOTS), round(rng.uniform(9, 11), 1), "standard", 133)),
        lambda: ("random", mk_random(round(rng.uniform(8, 12), 1), rng.choice(["standard", "premium", "cheap"]))),
        lambda: ("follow", mk_follower(round(rng.uniform(9, 12), 1), rng.choice(["standard", "premium"]))),
        lambda: ("greedy", mk_forecast_greedy(rng.choice(["standard", "premium"]), rng.uniform(4, 7))),
        lambda: ("undercut", mk_undercut()),
    ]
    return [rng.choice(pool)() for _ in range(k)]


def make_strategy_bot(params=None):
    import importlib
    mod = importlib.reload(S) if False else S

    def f(st, i, mk):
        old = dict(mod.PARAMS)
        if params:
            mod.PARAMS.update(params)
        try:
            if getattr(mk, "_mem_owner", None) is None:
                mk._mem_owner = mod.Memory()
            saved = mod.MEM
            mod.MEM = mk._mem_owner
            try:
                return mod.decide(st)
            finally:
                mod.MEM = saved
        finally:
            mod.PARAMS.clear(); mod.PARAMS.update(old)
    return f


def run_suite(bot_fn, seeds, rivals_list, jitter=1.0, include_self_copies=0):
    res = []
    for sd in seeds:
        for k in rivals_list:
            rng = random.Random(sd * 1000 + k)
            riv = rival_pool(rng, k)
            bots = [("me", bot_fn)] + riv
            mk = Market(sd * 7919 + k, bots, truth={"jitter": jitter})
            mk.run()
            res.append((sd, k, mk.cash[0], mk.stats[0]))
    return res


def summarize(name, res):
    p = [r[2] for r in res]
    print("%-28s mean %8.0f  median %8.0f  sd %7.0f  worst %8.0f" % (
        name, statistics.mean(p), statistics.median(p), statistics.pstdev(p), min(p)))
    by = {}
    for sd, k, c, _ in res:
        by.setdefault(k, []).append(c)
    print("    by rivals: " + "  ".join("%d:%.0f" % (k, statistics.mean(v)) for k, v in sorted(by.items())))
    return statistics.mean(p)


if __name__ == "__main__":
    seeds = list(range(1, 9))
    rivals = [3, 5, 8]
    summarize("starter", run_suite(starter, seeds, rivals))
    summarize("strategy", run_suite(make_strategy_bot(), seeds, rivals))
