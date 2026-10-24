"""A self-correcting version of the Brain that learns from the trucks that beat it. Python 3.8+, stdlib only.

Every night, before ordering, LearningBrain reviews the day that just finished:

  1. It finds its "teachers": the cash leader, every truck ahead of us in the standings, and any
     truck that made more than us yesterday.
  2. For each teacher it replays the teacher's choice (spot, quality, price) through the exact
     model we used the night before, with the weather that really happened. If the teacher really
     beat us by more than the model predicted, our model underrated their choice: that gap is a
     mistake to correct.
  3. It attributes the gap to what the teacher did differently and updates its lessons:
       - parked somewhere else      -> spot bias (separately for event days, e.g. "game day -> stadium")
       - served a different quality -> quality bias
       - same spot and quality, other price -> a price anchor at the price that worked
       - leaders with higher reputation -> more weight on reputation
     and from our own result: sold out with hungry customers left, or carried leftovers -> a demand
     correction for that spot, so meal counts follow.
  4. The lessons shift the score of every candidate order tomorrow, so the same mistake costs more
     and more until the bot stops making it. Lessons fade a little each day so stale ones go away.

Lessons and a day-by-day journal are saved to lessons.json, so learning carries over day after
day (48 days and beyond) and across markets and restarts.
"""
import json
import math
import os
import time

import strategy as S
from strategy import COST, INF, QUALITIES, REP_STEP, SPOTS, VALUE, Brain, _num, event_kinds, qindex, rep_factor

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MEMORY = os.path.join(HERE, "lessons.json")

LEARN = {
    "lr": 0.35,             # share of an unexplained profit gap turned into bias each day
    "decay": 0.96,          # daily fade of every lesson
    "new_market_keep": 0.5, # share of plain spot lessons kept when a new market starts (spots get new hidden crowds)
    "bias_cap": 80.0,       # max $ any single bias can add or remove from a candidate's score
    "gap_cap": 200.0,       # max $ of one teacher's gap counted in a day
    "min_gap": 8.0,         # gaps below this are noise
    "max_teachers": 4,
    "price_bonus": 40.0,    # $ bonus at full strength for pricing at a proven price
    "price_width": 1.0,     # $ distance over which the price bonus fades out
    "demand_rate": 0.3,     # EMA rate for the per-spot demand correction
    "rep_step": 0.15,       # change to rep_w when leaders out-reputation us
    "rep_w_min": 1.0,
    "rep_w_max": 6.0,
    "journal_keep": 400,    # journal entries kept on disk
}


def fresh_lessons():
    return {
        "version": 1,
        "days_learned": 0,
        "spot_bias": {s: 0.0 for s in SPOTS},
        "event_spot_bias": {},              # "game|stadium" -> $
        "qual_bias": {q: 0.0 for q in QUALITIES},
        "price_anchor": {},                 # "spot|quality" -> {"price": $, "strength": 0..1}
        "demand_mult": {s: 1.0 for s in SPOTS},
        "rep_w": S.PARAMS["rep_w"],
        "journal": [],
    }


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class LearningBrain(Brain):
    def __init__(self, params=None, seed=None, memory_path=DEFAULT_MEMORY, persist=True, verbose=True,
                 learn=None, lessons=None):
        self.L = dict(LEARN)
        if learn:
            self.L.update(learn)
        self.memory_path = memory_path
        self.persist = persist and bool(memory_path)
        self.verbose = verbose
        self.lessons = lessons if lessons is not None else self._load()
        self.ctx = {}
        super().__init__(params, seed)

    # ---- memory

    def _load(self):
        if self.memory_path and os.path.exists(self.memory_path):
            try:
                with open(self.memory_path) as f:
                    data = json.load(f)
                base = fresh_lessons()
                base.update(data)
                return base
            except (OSError, ValueError):
                pass
        return fresh_lessons()

    def save(self):
        if not self.persist:
            return
        self.lessons["journal"] = self.lessons["journal"][-self.L["journal_keep"]:]
        tmp = self.memory_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self.lessons, f, indent=1)
            os.replace(tmp, self.memory_path)
        except OSError:
            pass

    def say(self, msg):
        if self.verbose:
            print("  [learn] " + msg)

    # ---- hooks into the Brain

    def reset(self):
        super().reset()
        self.ctx = {}
        if getattr(self, "market", None) is not None and getattr(self, "lessons", None):
            # spot crowds are re-drawn per market; keep only part of what we learned about plain spots
            keep = self.L["new_market_keep"]
            for s in SPOTS:
                self.lessons["spot_bias"][s] *= keep
                self.lessons["demand_mult"][s] = 1.0 + (self.lessons["demand_mult"][s] - 1.0) * keep
            self.lessons["price_anchor"] = {}

    def observe(self, state, last):
        super().observe(state, last)
        try:
            self.review(state, last)
        except Exception as e:          # learning must never stop the bot from ordering
            self.say("review failed: %r" % (e,))

    def decide(self, state):
        self.p["rep_w"] = self.lessons["rep_w"]
        fc = state.get("forecast") or {}
        day = state.get("order_for_day") or 0
        self._today = {"day": day, "kinds": sorted(event_kinds(fc.get("events"))), "spots": {}}
        order = super().decide(state)
        if order is not None:
            key = (order["spot"], qindex(order["quality"]), round(order["price"], 2))
            self._today["order"] = dict(order)
            self._today["pred"] = self._today["spots"].get(order["spot"], {}).get("preds", {}).get(key)
        self._today["probs"] = dict(S.weather_probs(fc))
        for info in self._today["spots"].values():
            info.pop("preds", None)
        self.ctx = {day: self._today}
        return order

    def _price_grid(self, qi, rf, groups):
        prices = set(super()._price_grid(qi, rf, groups))
        cap = min(30.0, VALUE[qi] * rf)
        for key, a in self.lessons["price_anchor"].items():
            s, q = key.split("|")
            if qindex(q) == qi and a["strength"] > 0.05:
                for d in (-0.25, 0.0, 0.25):
                    p = round(a["price"] + d, 2)
                    if COST[qi] < p <= cap:
                        prices.add(p)
        return sorted(prices)

    def _evaluate(self, spot, qi, price, rf, rep, mu, sd, qmix, scen, groups, days_left):
        info = self._today["spots"].setdefault(spot, {})
        if "mu" not in info:
            info.update(rf=rf, rep=rep, mu=mu, sd=sd, qmix=list(qmix), scen=scen, days_left=days_left,
                        groups=[(gp, [list(r) for r in rv]) for gp, rv in groups], preds={})
        profit, meals, sold = self.value(spot, qi, price, rf, rep, mu, sd, qmix, scen, groups,
                                         self.lessons["demand_mult"][spot])
        rep_bonus = 0.0
        if meals > 0 and sold >= 1:
            new_rep = clamp(rep + REP_STEP[qi], 0.0, 100.0)
            h = min(days_left, self.p["rep_horizon"])
            rep_bonus = self.p["rep_w"] * (new_rep - rep) * 0.006 * 12.0 * 70.0 * h / max(self.p["rep_horizon"], 1)
        bias = self.bias(spot, qi, price, self._today["kinds"]) if meals > 0 else 0.0
        info["preds"][(spot, qi, round(price, 2))] = (profit, meals, sold)
        order = {"spot": spot, "price": round(price, 2), "quality": QUALITIES[qi], "meals": meals}
        return profit + rep_bonus + bias, order, profit

    # ---- the model, reusable for replays

    def value(self, spot, qi, price, rf, rep, mu, sd, qmix, scen, groups, dmult=1.0, exclude=None):
        """Newsvendor-optimal (expected profit, meals, expected sold) for an order under a model."""
        dist = []
        for gp, rivals in groups:
            rv = [r for r in rivals if r[0] != exclude]
            trucks = [(r[1], r[2], r[3]) for r in rv] + [(price, qi, rf)]
            caps = [r[4] for r in rv] + [INF]
            segs = S.build_segments(trucks, mu, sd, qmix)
            for sp, n in scen:
                dist.append((gp * sp, S.fluid(segs, caps, n)[-1] * dmult))
        noise = self.p["noise"]
        full = []
        for pr, d in dist:
            for f, w in ((1 - 1.4 * noise, 0.25), (1.0, 0.5), (1 + 1.4 * noise, 0.25)):
                full.append((pr * w, max(0.0, d * f)))
        max_meals = int(S.BUDGET // COST[qi] + 1e-9)
        cands = sorted(set(min(max_meals, int(d)) for _, d in full) | {min(max_meals, int(d) + 1) for _, d in full})
        best = (0.0, 0, 0.0)
        for m in cands:
            if m <= 0:
                continue
            es = sum(pr * min(m, d) for pr, d in full)
            v = price * es - COST[qi] * m
            if v > best[0]:
                best = (v, m, es)
        return best

    def bias(self, spot, qi, price, kinds):
        L, les = self.L, self.lessons
        b = les["spot_bias"].get(spot, 0.0) + les["qual_bias"].get(QUALITIES[qi], 0.0)
        for k in kinds:
            b += les["event_spot_bias"].get(k + "|" + spot, 0.0)
        a = les["price_anchor"].get(spot + "|" + QUALITIES[qi])
        if a and a["strength"] > 0:
            b += L["price_bonus"] * a["strength"] * max(0.0, 1.0 - abs(price - a["price"]) / L["price_width"])
        return clamp(b, -2 * L["bias_cap"], 2 * L["bias_cap"])

    # ---- the daily review

    def review(self, state, last):
        L, les = self.L, self.lessons
        day = last.get("day")
        ctx = self.ctx.get(day)
        costs = (state.get("rules") or {}).get("costs") or {}
        cost = lambda q: _num(costs.get(QUALITIES[q]), COST[q])
        weather = str(last.get("weather") or "clear").lower()
        kinds = sorted(event_kinds(last.get("events")))

        # everyone's results yesterday
        rows, spot_info = {}, {}
        for spot, info in S.spot_entries(last):
            if spot not in SPOTS:
                continue
            spot_info[spot] = info
            for t in info.get("trucks") or []:
                q = qindex(t.get("quality"))
                price, meals, sold = _num(t.get("price")), _num(t.get("meals")), _num(t.get("sold"))
                rows[S.truck_name(t)] = {"name": t.get("truck_name") or S.truck_name(t), "spot": spot, "q": q,
                                         "price": price, "meals": meals, "sold": sold,
                                         "profit": sold * price - meals * cost(q)}
        me = self.me or S.truck_name(state.get("you") or {})
        you = last.get("you") or {}
        mine = rows.get(me)
        my_profit = _num(you.get("profit"), mine["profit"] if mine else 0.0)

        # standings
        cash = {S.truck_name(t): _num(t.get("cash")) for t in state.get("trucks") or []}
        my_cash = _num((state.get("you") or {}).get("cash"), cash.get(me, 0.0))
        leader = max(cash, key=cash.get) if cash else None
        ahead = {n for n, c in cash.items() if n != me and c > my_cash}

        teachers = [n for n, r in rows.items() if n != me and r["profit"] > my_profit + L["min_gap"]]
        teachers.sort(key=lambda n: (n != leader, n not in ahead, -rows[n]["profit"]))
        teachers = teachers[:L["max_teachers"]]

        notes = []
        for k in les["spot_bias"]:
            les["spot_bias"][k] *= L["decay"]
        for d in (les["event_spot_bias"], ):
            for k in list(d):
                d[k] *= L["decay"]
                if abs(d[k]) < 0.5:
                    del d[k]
        for k in les["qual_bias"]:
            les["qual_bias"][k] *= L["decay"]
        for k in list(les["price_anchor"]):
            les["price_anchor"][k]["strength"] *= L["decay"]

        # 1) learn from every truck that did better than us
        for name in teachers:
            r = rows[name]
            weight = 1.5 if name == leader else (1.2 if name in ahead else 1.0)
            if ctx is None or mine is None or "order" not in ctx:
                notes.append("%s made $%.0f at %s (%s @ $%.2f) while we %s" % (
                    r["name"], r["profit"], r["spot"], QUALITIES[r["q"]], r["price"],
                    "sat out" if mine is None else "had no model context"))
                if mine is None and r["profit"] > 50:
                    self._shift_spot(r["spot"], kinds, L["lr"] * min(r["profit"], L["gap_cap"]) * 0.3 * weight)
                continue
            pred_t = self._replay(ctx, r["spot"], r["q"], r["price"], weather, exclude=name, rep=self._rep_of(name, r["spot"], state, before=True))
            pred_me = self._replay(ctx, mine["spot"], mine["q"], mine["price"], weather)
            if pred_t is None or pred_me is None:
                continue
            gap = (r["profit"] - my_profit) - (pred_t - pred_me)
            gap = clamp(gap, -L["gap_cap"], L["gap_cap"])
            if gap < L["min_gap"]:
                notes.append("%s beat us by $%.0f but the model saw it coming (luck/risk), no change" % (
                    r["name"], r["profit"] - my_profit))
                continue
            step = L["lr"] * gap * weight / max(1, len(teachers))
            what = []
            if r["spot"] != mine["spot"]:
                self._shift_spot(r["spot"], kinds, step / 2)
                self._shift_spot(mine["spot"], kinds, -step / 2)
                what.append("spot %s +$%.0f / %s -$%.0f%s" % (r["spot"], step / 2, mine["spot"], step / 2,
                                                             (" on " + "+".join(kinds)) if kinds else ""))
            if r["q"] != mine["q"]:
                qt, qm = QUALITIES[r["q"]], QUALITIES[mine["q"]]
                les["qual_bias"][qt] = clamp(les["qual_bias"][qt] + step / 3, -L["bias_cap"], L["bias_cap"])
                les["qual_bias"][qm] = clamp(les["qual_bias"][qm] - step / 3, -L["bias_cap"], L["bias_cap"])
                what.append("quality %s over %s" % (qt, qm))
            if r["spot"] == mine["spot"] and r["q"] == mine["q"] and abs(r["price"] - mine["price"]) > 0.1:
                key = r["spot"] + "|" + QUALITIES[r["q"]]
                a = les["price_anchor"].get(key, {"price": r["price"], "strength": 0.0})
                s_add = min(1.0, gap / 150.0) * weight * 0.5
                tot = a["strength"] + s_add
                a["price"] = round((a["price"] * a["strength"] + r["price"] * s_add) / max(tot, 1e-9), 2)
                a["strength"] = min(1.0, tot)
                les["price_anchor"][key] = a
                what.append("price anchor %s %s ~$%.2f (%.2f)" % (r["spot"], QUALITIES[r["q"]], a["price"], a["strength"]))
            notes.append("%s%s made $%.0f at %s (%s @ $%.2f) vs our $%.0f at %s (%s @ $%.2f); model underrated it by $%.0f -> %s" % (
                r["name"], " (leader)" if name == leader else "", r["profit"], r["spot"], QUALITIES[r["q"]], r["price"],
                my_profit, mine["spot"], QUALITIES[mine["q"]], mine["price"], gap, "; ".join(what) or "no difference to fix"))

        # 2) learn from our own sales: under- or over-stocked versus the model
        if mine is not None and ctx is not None and ctx.get("pred"):
            _, meals_pred, sold_pred = ctx["pred"]
            hungry = _num((spot_info.get(mine["spot"]) or {}).get("hungry"), 0)
            dm = les["demand_mult"][mine["spot"]]
            if sold_pred > 1:
                if mine["sold"] >= mine["meals"] - 0.5 and hungry > 0:
                    ratio = max(1.0, (mine["sold"] + min(hungry, mine["sold"] * 0.3)) / sold_pred)
                    reason = "sold out with %d hungry left" % hungry
                else:
                    ratio = mine["sold"] / sold_pred
                    reason = "sold %d of %d (model expected %.0f)" % (mine["sold"], mine["meals"], sold_pred)
                new = clamp(dm * (1 - L["demand_rate"]) + dm * ratio * L["demand_rate"], 0.7, 1.4)
                les["demand_mult"][mine["spot"]] = round(new, 4)
                if abs(new - dm) > 0.01:
                    notes.append("%s: demand correction at %s %.2f -> %.2f" % (reason, mine["spot"], dm, new))

        # 3) reputation: if the trucks ahead of us hold more reputation, value it more
        reps = {S.truck_name(t): _num(t.get("reputation_avg"), 50) for t in state.get("trucks") or []}
        if ahead and me in reps:
            gap_rep = sum(reps.get(n, 50) for n in ahead) / len(ahead) - reps[me]
            if gap_rep > 5:
                les["rep_w"] = round(clamp(les["rep_w"] + L["rep_step"], L["rep_w_min"], L["rep_w_max"]), 3)
                notes.append("trucks ahead average %.0f more reputation -> rep_w %.2f" % (gap_rep, les["rep_w"]))

        les["days_learned"] += 1
        entry = {"t": int(time.time()), "mode": state.get("mode"), "market": state.get("market"), "day": day,
                 "weather": weather, "events": kinds, "our_profit": round(my_profit, 2),
                 "our_cash": round(my_cash, 2), "leader": rows.get(leader, {}).get("name", leader),
                 "rank": 1 + len(ahead), "lessons": notes}
        les["journal"].append(entry)
        if notes:
            self.say("day %s (rank %d): %s" % (day, 1 + len(ahead), " | ".join(notes)))
        self.save()

    def _shift_spot(self, spot, kinds, amount):
        L, les = self.L, self.lessons
        if kinds:
            for k in kinds:
                key = k + "|" + spot
                les["event_spot_bias"][key] = clamp(les["event_spot_bias"].get(key, 0.0) + amount / len(kinds),
                                                    -L["bias_cap"], L["bias_cap"])
        else:
            les["spot_bias"][spot] = clamp(les["spot_bias"][spot] + amount, -L["bias_cap"], L["bias_cap"])

    def _replay(self, ctx, spot, qi, price, weather, exclude=None, rep=None):
        """Expected profit of an order under last night's model, with the weather that really happened."""
        info = ctx["spots"].get(spot)
        if not info:
            return None
        probs = ctx.get("probs") or {"clear": 1.0}
        wm = lambda w: S.weather_mult(w, spot) * self.wfac.get((spot, w), 1.0)
        avg = sum(p * wm(w) for w, p in probs.items()) or 1.0
        expected = sum(p * n for p, n in info["scen"])
        scen = [(1.0, expected * wm(weather) / avg)]
        rf = info["rf"] if rep is None else rep_factor(rep)
        groups = [(gp, [tuple(r) for r in rv]) for gp, rv in info["groups"]]
        return self.value(spot, qi, price, rf, info["rep"], info["mu"], info["sd"], tuple(info["qmix"]),
                          scen, groups, self.lessons["demand_mult"][spot], exclude=exclude)[0]


def summary(lessons):
    """Human-readable view of what has been learned so far."""
    out = ["days learned: %d   rep_w: %.2f" % (lessons["days_learned"], lessons["rep_w"])]
    out.append("spot bias:   " + ", ".join("%s %+.0f" % (s, v) for s, v in lessons["spot_bias"].items()))
    if lessons["event_spot_bias"]:
        out.append("event bias:  " + ", ".join("%s %+.0f" % (k, v) for k, v in sorted(
            lessons["event_spot_bias"].items(), key=lambda kv: -abs(kv[1]))[:10]))
    out.append("quality:     " + ", ".join("%s %+.0f" % (q, v) for q, v in lessons["qual_bias"].items()))
    out.append("demand:      " + ", ".join("%s x%.2f" % (s, v) for s, v in lessons["demand_mult"].items()))
    for k, a in lessons["price_anchor"].items():
        if a["strength"] > 0.05:
            out.append("price:       %s ~$%.2f (strength %.2f)" % (k, a["price"], a["strength"]))
    return "\n".join(out)
