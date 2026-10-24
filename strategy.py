"""Decision engine for the Night Shift food-truck game. Standard library only, Python 3.8+.

The bot calls Brain.decide(state) once per night. The brain:
  1. observes last_day: tracks every rival (where they parked, price, quality, meals, sales) and
     their reputation per spot, calibrates the crowd forecast, and fits the hidden customer
     distributions (price limits and quality minimums) per spot to what actually sold;
  2. forecasts tomorrow's customers per spot as a few weather scenarios;
  3. predicts where rivals park by sampling from their history;
  4. for every spot x quality x price, runs a fluid version of the customer-choice rules
     (including sell-outs spilling to second choices) to get our demand distribution;
  5. sizes meals with a newsvendor rule and adds the long-term value of the reputation change;
  6. returns the best order.
"""
import math
import random
import statistics
import time

QUALITIES = ("cheap", "standard", "premium")
COST = (2.50, 3.75, 5.50)
VALUE = (8.0, 12.0, 16.0)
REP_STEP = (-3, 2, 5)
SPOTS = ("oval", "thompson", "union", "stadium", "highst", "northres")
BUDGET = 500.0
INF = float("inf")

# Prior beliefs per spot: mean price limit, spread, share whose minimum quality is
# cheap / standard / premium, and share of campus customers on an ordinary clear day.
PRIORS = {
    "highst":   (12.5, 5.0, (0.50, 0.35, 0.15), 0.24),
    "oval":     (13.0, 3.0, (0.40, 0.40, 0.20), 0.20),
    "union":    (13.0, 3.0, (0.20, 0.60, 0.20), 0.18),
    "northres": (11.0, 2.5, (0.50, 0.42, 0.08), 0.16),
    "thompson": (12.0, 3.0, (0.47, 0.40, 0.13), 0.14),
    "stadium":  (13.0, 3.0, (0.25, 0.50, 0.25), 0.08),
}

# Crowd multipliers by weather relative to a clear day, fitted to practice data (learned per spot
# as the market goes on). Rain empties the outdoor spots most.
WEATHER_MULT = {
    "cloudy": {"oval": 1.00, "stadium": 0.98, "highst": 0.95, "northres": 0.94, "thompson": 0.96, "union": 0.95},
    "cold":   {"oval": 0.79, "stadium": 0.78, "highst": 0.85, "northres": 0.90, "thompson": 0.89, "union": 0.90},
    "rain":   {"oval": 0.53, "stadium": 0.55, "highst": 0.71, "northres": 0.75, "thompson": 0.75, "union": 0.75},
    "storm":  {"oval": 0.45, "stadium": 0.45, "highst": 0.65, "northres": 0.70, "thompson": 0.70, "union": 0.70},
    "snow":   {"oval": 0.50, "stadium": 0.50, "highst": 0.70, "northres": 0.75, "thompson": 0.75, "union": 0.75},
}

# Event effects: crowd multipliers (only used when the forecast has no expected_customers),
# shift of the mean price limit, and an override of the quality-minimum mix.
EVENTS = {
    "game":     {"crowd": {"stadium": 5.0, "highst": 1.5, "*": 1.2}, "mu": {"stadium": 1.0, "*": 0.5},
                 "qmix": {"stadium": (0.45, 0.40, 0.15)}},
    "midterm":  {"crowd": {"thompson": 1.7}, "mu": {"thompson": 0.5}, "qmix": {"thompson": (0.10, 0.35, 0.55)}},
    "concert":  {"crowd": {"union": 1.8}, "mu": {"union": 1.0}},
    "late":     {"crowd": {"highst": 2.0, "northres": 2.0, "*": 0.5},
                 "qmix": {"highst": (0.75, 0.20, 0.05), "northres": (0.75, 0.20, 0.05)}},
    "rally":    {"crowd": {"oval": 2.0}},
}

# Tunable parameters (tuned offline with sim.py).
PARAMS = {
    "rep_w": 3.0,         # weight on the future value of a reputation point
    "rep_horizon": 12,    # days of future benefit counted for a reputation change
    "stay": 0.65,         # probability a rival parks where it parked yesterday (blended with history)
    "smart": 0.15,        # probability a rival heads for the busiest spot instead
    "noise": 0.12,        # relative day-to-day noise in a spot's demand
    "samples": 12,        # rival placement samples
    "learn_temp": 6.0,    # temperature of the likelihood used to fit customer distributions
    "time_budget": 0.9,   # seconds allowed for the search
    "min_profit": 1.0,    # sit out unless the order is expected to make at least this much
    "price_sd": 1.0,      # default day-to-day price spread assumed for adaptive rivals
    "undercut": 0.25,     # always price at least this far below a watched rival of equal or better quality
    "watch_prob": 0.25,   # watch a rival at a spot if it parked there yesterday or is this likely to show up
    "min_margin": 0.5,    # never undercut below cost plus this much
}

# Grid of corrections to the prior customer model that the learner weighs per spot.
LEARN_GRID = [(dmu, sds, qs)
              for dmu in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
              for sds in (0.6, 1.0, 1.5)
              for qs in (-0.15, 0.0, 0.15)]


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x * 0.7071067811865476))


def qindex(q):
    try:
        return QUALITIES.index(str(q).lower())
    except ValueError:
        return 1


def rep_factor(rep):
    return 0.7 + 0.6 * max(0.0, min(100.0, rep)) / 100.0


def shift_qmix(qmix, s):
    """Move mass s from 'accepts cheap' to 'wants premium' (negative s goes the other way)."""
    c, st, p = qmix
    s = max(-p, min(c, s))
    return (c - s, st, p + s)


# ---------------------------------------------------------------- customer model

def build_segments(trucks, mu, sd, qmix):
    """Split customers into segments that share the same ordered list of acceptable trucks.

    trucks: list of (price, quality index, reputation factor). Returns [(mass, [truck idx...])],
    each list ordered by the customer's preference (highest value first, ties to the cheaper
    truck, and the last truck in `trucks` loses exact ties so we never count on winning one).
    """
    segs = []
    sd = max(sd, 0.3)
    for k in range(3):
        pk = qmix[k]
        if pk <= 0:
            continue
        elig = [i for i, t in enumerate(trucks) if t[1] >= k and VALUE[t[1]] * t[2] - t[0] >= 0]
        if not elig:
            continue
        prices = sorted(set(trucks[i][0] for i in elig))
        for j, lo in enumerate(prices):
            top = norm_cdf((prices[j + 1] - mu) / sd) if j + 1 < len(prices) else 1.0
            mass = pk * (top - norm_cdf((lo - mu) / sd))
            if mass <= 1e-9:
                continue
            feas = [i for i in elig if trucks[i][0] <= lo]
            feas.sort(key=lambda i: (-(VALUE[trucks[i][1]] * trucks[i][2] - trucks[i][0]), trucks[i][0], i))
            segs.append((mass, feas))
    return segs


def fluid(segs, caps, n_customers):
    """Expected meals sold by each truck when n_customers arrive evenly over the day.

    Customers flow to their top truck that still has food; when a truck runs out its later
    customers move to their next choice. caps: meals per truck (INF = never runs out).
    """
    n = len(caps)
    sold = [0.0] * n
    rem = list(caps)
    out = [False] * n
    t = 1.0
    while t > 1e-9:
        rate = [0.0] * n
        for mass, feas in segs:
            for i in feas:
                if not out[i]:
                    rate[i] += mass * n_customers
                    break
        dt, hit = t, -1
        for i in range(n):
            if rate[i] > 0 and rem[i] < INF:
                d = rem[i] / rate[i]
                if d < dt:
                    dt, hit = d, i
        for i in range(n):
            if rate[i] > 0:
                x = rate[i] * dt
                sold[i] += x
                rem[i] -= x
        t -= dt
        if hit < 0:
            break
        out[hit] = True
    return sold


# ---------------------------------------------------------------- state parsing helpers

def _num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def event_names(events):
    """Normalize an events field (list of strings/dicts, dict, or string) to lowercase strings."""
    out = []
    if not events:
        return out
    if isinstance(events, dict):
        events = [dict(v, key=k) if isinstance(v, dict) else k for k, v in events.items()]
    if isinstance(events, str):
        events = [events]
    for e in events:
        if isinstance(e, dict):
            out.append(" ".join(str(v) for v in e.values() if isinstance(v, str)).lower())
        else:
            out.append(str(e).lower())
    return out


def event_kinds(events):
    """Map announced events to EVENTS keys (the server sends keys like 'gameday', 'latenight')."""
    if isinstance(events, list) and events and all(isinstance(e, dict) and e.get("key") for e in events):
        events = [e["key"] for e in events]
    kinds = set()
    for e in event_names(events):
        for k in EVENTS:
            if k in e:
                kinds.add(k)
        if "rush" in e or "night" in e:
            kinds.add("late")
        if "exam" in e or "final" in e:
            kinds.add("midterm")
        if "football" in e or "stadium" in e:
            kinds.add("game")
    return kinds


def spot_entries(last_day):
    """Yield (spot, info) from last_day, whether spots are a dict or a list."""
    spots = last_day.get("spots")
    if spots is None:
        spots = {k: v for k, v in last_day.items() if k in SPOTS}
    if isinstance(spots, dict):
        for k, v in spots.items():
            if isinstance(v, dict):
                yield k, v
    elif isinstance(spots, list):
        for v in spots:
            if isinstance(v, dict):
                yield v.get("spot") or v.get("name") or v.get("id"), v


def truck_name(t):
    return str(t.get("id") or t.get("truck_name") or t.get("name") or t.get("truck") or t.get("id") or t.get("company") or "?")


def weather_probs(forecast):
    w = forecast.get("weather") or forecast.get("weather_odds") or forecast.get("weather_probs") or {}
    if isinstance(w, dict):
        probs = {str(k).lower(): _num(v) for k, v in w.items() if _num(v) > 0}
    elif isinstance(w, str):
        probs = {w.lower(): 1.0}
    else:
        probs = {}
    tot = sum(probs.values())
    if tot <= 0:
        return {"clear": 1.0}
    if tot > 1.5:                    # percentages
        probs = {k: v / tot for k, v in probs.items()}
    return probs


def weather_mult(w, spot):
    for k, m in WEATHER_MULT.items():
        if k in w:
            return m.get(spot, 1.0)
    return 1.0


# ---------------------------------------------------------------- the brain

class Brain:
    def __init__(self, params=None, seed=None):
        self.p = dict(PARAMS)
        if params:
            self.p.update(params)
        self.seed = seed
        self.market = None
        self.reset()

    def reset(self):
        self.seen_day = None
        self.last_order_day = None
        self.rivals = {}          # name -> list of dicts (day, spot, price, q, meals, sold)
        self.reps = {}            # name -> {spot: rep}
        self.logw = {s: [0.0] * len(LEARN_GRID) for s in SPOTS}
        self.wfac = {}            # (spot, weather) -> learned crowd correction
        self.cal = {s: 1.0 for s in SPOTS}   # forecast -> actual correction
        self.pending_forecast = None          # (day, {spot: expected}, weather probs)
        self.me = None
        self.days_at = {s: 0 for s in SPOTS}
        self.rival_note = ""

    # ---- bookkeeping

    def _market_key(self, state):
        return (state.get("mode"), state.get("market"))

    def sync(self, state):
        key = self._market_key(state)
        day = state.get("order_for_day")
        if key != self.market or (day is not None and self.last_order_day is not None
                                  and day < self.last_order_day) or day in (0, 1) and self.seen_day:
            self.market = key
            self.reset()
        if day is not None:
            self.last_order_day = day
        last = state.get("last_day") or {}
        ld = last.get("day")
        if last and ld is not None and ld != self.seen_day:
            self.seen_day = ld
            try:
                self.observe(state, last)
            except Exception:
                pass

    def observe(self, state, last):
        you = last.get("you") or {}
        me_name = truck_name(state.get("you") or {}) if state.get("you") else None
        weather = str(last.get("weather") or "clear").lower()
        if isinstance(last.get("weather"), dict):
            weather = str(last["weather"].get("kind") or last["weather"].get("type") or "clear").lower()
        kinds = event_kinds(last.get("events"))
        visited = {}
        for spot, info in spot_entries(last):
            if spot not in SPOTS:
                continue
            trucks = info.get("trucks") or []
            n_cust = _num(info.get("customers"), -1)
            hungry = _num(info.get("hungry"), -1)
            obs = []
            for t in trucks:
                name = truck_name(t)
                rec = {"day": last.get("day"), "spot": spot, "price": _num(t.get("price"), 10),
                       "q": qindex(t.get("quality")), "meals": _num(t.get("meals")), "sold": _num(t.get("sold"))}
                mine = (me_name and name == me_name) or (
                    you.get("spot") == spot and abs(_num(you.get("price")) - rec["price"]) < 1e-6
                    and qindex(you.get("quality")) == rec["q"] and _num(you.get("meals")) == rec["meals"])
                if mine:
                    self.me = name
                    rec["me"] = True
                else:
                    self.rivals.setdefault(name, []).append(rec)
                visited[name] = rec
                obs.append((name, rec))
            if n_cust > 0 and obs:
                self._learn(spot, kinds, n_cust, hungry, obs, state)
            # crowd calibration against what we forecast for this day
            if self.pending_forecast and self.pending_forecast[0] == last.get("day") and n_cust >= 0:
                exp = self.pending_forecast[1].get(spot)
                probs = self.pending_forecast[2]
                if exp and exp > 0:
                    avg = sum(p * weather_mult(w, spot) * self.wfac.get((spot, w), 1.0) for w, p in probs.items())
                    pred = exp * weather_mult(weather, spot) * self.wfac.get((spot, weather), 1.0) / max(avg, 1e-6)
                    ratio = max(0.4, min(2.5, n_cust / max(pred, 1.0)))
                    key = (spot, weather)
                    self.wfac[key] = 0.6 * self.wfac.get(key, 1.0) + 0.4 * self.wfac.get(key, 1.0) * ratio
        # rival reputation tracking: sold >= 1 moves rep at that spot, other spots drift toward 50
        for name, rec in visited.items():
            reps = self.reps.setdefault(name, {s: 50.0 for s in SPOTS})
            for s in SPOTS:
                if s == rec["spot"]:
                    if rec["sold"] >= 1:
                        reps[s] = max(0.0, min(100.0, reps[s] + REP_STEP[rec["q"]]))
                else:
                    reps[s] += 0.03 * (50.0 - reps[s])
        for name, reps in self.reps.items():
            if name not in visited:
                for s in SPOTS:
                    reps[s] += 0.03 * (50.0 - reps[s])
        # the server publishes every truck's reputation; prefer it over our own tracking
        for t in state.get("trucks") or []:
            if isinstance(t.get("reputation"), dict):
                reps = self.reps.setdefault(truck_name(t), {s: 50.0 for s in SPOTS})
                for s in SPOTS:
                    if s in t["reputation"]:
                        reps[s] = _num(t["reputation"][s], reps[s])

    def spot_model(self, spot, kinds, correction=None):
        """Customer model (mu, sd, qmix) at a spot under today's events and learned corrections."""
        mu, sd, qmix, _ = PRIORS[spot]
        for k in kinds:
            ev = EVENTS.get(k, {})
            mu += ev.get("mu", {}).get(spot, ev.get("mu", {}).get("*", 0.0))
            if spot in ev.get("qmix", {}):
                qmix = ev["qmix"][spot]
        if correction is None:
            correction = self.posterior(spot)
        dmu, sds, qs = correction
        return mu + dmu, sd * sds, shift_qmix(qmix, qs)

    def posterior(self, spot):
        lw = self.logw[spot]
        m = max(lw)
        ws = [math.exp(x - m) for x in lw]
        tot = sum(ws)
        dmu = sum(w * g[0] for w, g in zip(ws, LEARN_GRID)) / tot
        sds = sum(w * g[1] for w, g in zip(ws, LEARN_GRID)) / tot
        qs = sum(w * g[2] for w, g in zip(ws, LEARN_GRID)) / tot
        return dmu, sds, qs

    def _learn(self, spot, kinds, n_cust, hungry, obs, state):
        """Re-weight the candidate customer models by how well each predicts what sold."""
        trucks, caps, sold = [], [], []
        for name, rec in obs:
            rep = self._rep_of(name, spot, state, before=True)
            trucks.append((rec["price"], rec["q"], rep_factor(rep)))
            caps.append(rec["meals"] if rec["meals"] > 0 else INF)
            sold.append(rec["sold"])
        temp = self.p["learn_temp"]
        lw = self.logw[spot]
        for gi, g in enumerate(LEARN_GRID):
            mu, sd, qmix = self.spot_model(spot, kinds, g)
            pred = fluid(build_segments(trucks, mu, sd, qmix), caps, n_cust)
            loss = 0.0
            for ps, os_, cap in zip(pred, sold, caps):
                if cap < INF and os_ >= cap - 0.5 and ps >= cap - 0.5:
                    continue
                loss += (ps - os_) ** 2 / (os_ + 8.0)
            if hungry >= 0:
                ph = n_cust - sum(pred)
                loss += 0.5 * (ph - hungry) ** 2 / (hungry + 8.0)
            lw[gi] -= loss / temp
        m = max(lw)
        for i in range(len(lw)):
            lw[i] = max(lw[i] - m, -60.0)

    def _rep_of(self, name, spot, state, before=False):
        if name == self.me:
            reps = (state.get("you") or {}).get("reputation") or {}
            if not before and spot in reps:
                return _num(reps[spot], 50)
        r = self.reps.get(name)
        if r is None:
            return 50.0
        return r.get(spot, 50.0)

    # ---- forecasting

    def forecast_customers(self, state, kinds):
        fc = state.get("forecast") or {}
        exp = fc.get("expected_customers") or {}
        out = {}
        n_trucks = max(4, len(state.get("trucks") or []))
        for s in SPOTS:
            if isinstance(exp, dict) and _num(exp.get(s), 0) > 0:
                out[s] = _num(exp[s])
            else:
                base = 110.0 * n_trucks * PRIORS[s][3]
                for k in kinds:
                    cr = EVENTS.get(k, {}).get("crowd", {})
                    base *= cr.get(s, cr.get("*", 1.0))
                out[s] = base
        return out

    def scenarios(self, spot, expected, probs):
        """Demand scenarios [(prob, customers)] for a spot from the weather odds."""
        avg = sum(p * weather_mult(w, spot) * self.wfac.get((spot, w), 1.0) for w, p in probs.items())
        out = []
        for w, p in probs.items():
            if p < 0.03:
                continue
            m = weather_mult(w, spot) * self.wfac.get((spot, w), 1.0) / max(avg, 1e-6)
            out.append((p, expected * m))
        tot = sum(p for p, _ in out) or 1.0
        merged = {}
        for p, n in out:
            k = round(n, 1)
            merged[k] = merged.get(k, 0.0) + p / tot
        return [(p, n) for n, p in merged.items()]

    @staticmethod
    def spot_groups(samples, spot):
        """Merge identical rival line-ups at a spot: [(probability, rivals)]."""
        groups = {}
        for smp in samples:
            key = tuple(sorted((x[0], x[1]) for x in smp[spot]))
            groups.setdefault(key, [0, smp[spot]])[0] += 1
        return [(c / len(samples), rv) for c, rv in groups.values()]

    def rival_predictions(self, state, expected, day):
        """Sampled joint rival placements: list of {spot: [(name, price, q, rf, meals)]}."""
        rng = random.Random(hash((self.seed, self.market, day)) & 0xffffffff)
        busiest = max(expected, key=expected.get) if expected else None
        names = list(self.rivals)
        plans = []
        for name in names:
            hist = self.rivals[name][-10:]
            last = hist[-1]
            freq = {}
            for age, rec in enumerate(reversed(hist)):
                freq[rec["spot"]] = freq.get(rec["spot"], 0.0) + 0.75 ** age
            tot = sum(freq.values())
            # each rival's own habit: how often it parked where it parked the day before
            moves = [a["spot"] == b["spot"] for a, b in zip(hist, hist[1:])]
            prior = 2.0
            stay = (sum(moves) + prior * self.p["stay"]) / (len(moves) + prior)
            stay = min(0.95, stay)
            smart = self.p["smart"] * (1.0 - stay)
            probs = {s: (1 - stay - smart) * freq.get(s, 0.0) / tot for s in SPOTS}
            probs[last["spot"]] += stay
            if busiest:
                probs[busiest] += smart
            # price habit: steady rivals repeat their price; adaptive rivals get a spread
            prices = [r["price"] for r in hist]
            steps = [b - a for a, b in zip(prices, prices[1:])]
            if len(hist) >= 3 and max(prices) - min(prices) < 0.01 and len(set(r["q"] for r in hist)) == 1:
                psd, trend = 0.0, 0.0
            else:
                psd = statistics.pstdev(steps) if len(steps) >= 2 else self.p["price_sd"]
                psd = max(0.15, min(2.0, 0.5 * psd + 0.5 * self.p["price_sd"]))
                trend = 0.5 * sum(steps[-3:]) / len(steps[-3:]) if steps else 0.0
            plans.append((name, last, probs, psd, trend))
        samples = []
        for _ in range(int(self.p["samples"])):
            by_spot = {s: [] for s in SPOTS}
            for name, last, probs, psd, trend in plans:
                r, acc, pick = rng.random(), 0.0, last["spot"]
                for s in SPOTS:
                    acc += probs[s]
                    if r <= acc:
                        pick = s
                        break
                rep = self.reps.get(name, {}).get(pick, 50.0)
                price = last["price"]
                if psd > 0:
                    price = round(max(COST[last["q"]] + 0.1, price + trend + rng.gauss(0.0, psd)), 2)
                by_spot[pick].append((name, price, last["q"], rep_factor(rep),
                                      last["meals"] if last["meals"] > 0 else INF))
            samples.append(by_spot)
        return samples

    # ---- decision

    def decide(self, state):
        t0 = time.time()
        self.sync(state)
        p = self.p
        fc = state.get("forecast") or {}
        kinds = event_kinds(fc.get("events"))
        probs = weather_probs(fc)
        day = state.get("order_for_day") or 0
        days = state.get("days") or 30
        days_left = max(0, int(days) - int(day))
        expected = self.forecast_customers(state, kinds)
        self.pending_forecast = (day, dict(expected), dict(probs))
        my_reps = (state.get("you") or {}).get("reputation") or {}
        samples = self.rival_predictions(state, expected, day)
        watch = self.rival_watch(samples)

        best = None
        cands = []
        for spot in SPOTS:
            rep = _num(my_reps.get(spot, 50), 50)
            rf = rep_factor(rep)
            mu, sd, qmix = self.spot_model(spot, kinds)
            scen = self.scenarios(spot, expected[spot], probs)
            groups = self.spot_groups(samples, spot)
            for qi in range(3):
                ceil = self.price_ceiling(watch, spot, qi)
                if ceil is not None and ceil <= COST[qi] + p["min_margin"]:
                    continue   # can't undercut the rivals here at this quality and still make money
                prices = self._price_grid(qi, rf, groups)
                if ceil is not None:
                    prices = sorted(set(x for x in prices if x <= ceil) | {ceil})
                for price in prices:
                    res = self._evaluate(spot, qi, price, rf, rep, mu, sd, qmix, scen, groups, days_left)
                    cands.append(res)
                    if best is None or res[0] > best[0]:
                        best = res
            if time.time() - t0 > p["time_budget"] * 0.7:
                break
        # refine the price around the top few (spot, quality) choices
        cands.sort(key=lambda r: -r[0])
        done = set()
        for res in cands[:6]:
            spot, qi = res[1]["spot"], qindex(res[1]["quality"])
            if (spot, qi) in done:
                continue
            done.add((spot, qi))
            rep = _num(my_reps.get(spot, 50), 50)
            rf = rep_factor(rep)
            mu, sd, qmix = self.spot_model(spot, kinds)
            scen = self.scenarios(spot, expected[spot], probs)
            groups = self.spot_groups(samples, spot)
            base = res[1]["price"]
            ceil = self.price_ceiling(watch, spot, qi)
            for k in range(-10, 11):
                price = round(base + 0.05 * k, 2)
                if price <= COST[qi] or price > min(30.0, VALUE[qi] * rf):
                    continue
                if ceil is not None and price > ceil:
                    continue
                r2 = self._evaluate(spot, qi, price, rf, rep, mu, sd, qmix, scen, groups, days_left)
                if r2[0] > best[0]:
                    best = r2
            if time.time() - t0 > p["time_budget"]:
                break
        self.rival_note = self._rival_note(watch, best[1]["spot"]) if best else ""
        if best is None or best[2] < p["min_profit"]:
            return None
        self.days_at[best[1]["spot"]] += 1
        return best[1]

    def rival_watch(self, samples):
        """{spot: [(name, price, q)]}: rivals that parked there yesterday or are likely to show up."""
        n = max(1, len(samples))
        watch = {s: [] for s in SPOTS}
        for name, hist in self.rivals.items():
            last = hist[-1]
            if self.seen_day is not None and last.get("day") != self.seen_day:
                last = None   # it didn't play yesterday; rely on the predicted placements only
            for s in SPOTS:
                share = sum(1 for smp in samples if any(r[0] == name for r in smp[s])) / n
                if (last and last["spot"] == s) or share >= self.p["watch_prob"]:
                    ref = last or hist[-1]
                    watch[s].append((name, ref["price"], ref["q"]))
        return watch

    def price_ceiling(self, watch, spot, qi):
        """Highest price allowed at this spot and quality: just under every equal-or-better rival."""
        rivals = [rp for (_, rp, rq) in watch.get(spot, []) if rq >= qi]
        if not rivals:
            return None
        return round(min(rivals) - self.p["undercut"], 2)

    def _rival_note(self, watch, spot):
        rivals = watch.get(spot, [])
        if not rivals:
            return "no rivals expected at %s" % spot
        return "RIVAL WATCH at %s: " % spot + ", ".join(
            "%s (%s $%.2f)" % (name, QUALITIES[q], price) for name, price, q in rivals)

    def _price_grid(self, qi, rf, groups):
        cap = min(30.0, VALUE[qi] * rf)
        lo = COST[qi] + 0.5
        prices = set()
        x = lo
        while x <= cap + 1e-9:
            prices.add(round(x, 2))
            x += 0.5
        prices.add(round(cap - 0.01, 2))
        # just undercut each rival's customer value so we win their customers' first choice
        for _, rivals in groups:
            for (_, rp, rq, rrf, _) in rivals:
                b = VALUE[qi] * rf - (VALUE[rq] * rrf - rp) - 0.01
                if lo - 0.5 < b <= cap:
                    prices.add(round(b, 2))
                if lo - 0.5 < rp - 0.01 <= cap:
                    prices.add(round(rp - 0.01, 2))
        return sorted(p for p in prices if p > COST[qi])

    def _evaluate(self, spot, qi, price, rf, rep, mu, sd, qmix, scen, groups, days_left):
        """Expected profit (plus reputation value) of an order; returns (score, order, profit)."""
        dist = []   # (prob, demand)
        for gp, rivals in groups:
            trucks = [(r[1], r[2], r[3]) for r in rivals] + [(price, qi, rf)]
            caps = [r[4] for r in rivals] + [INF]
            segs = build_segments(trucks, mu, sd, qmix)
            for sp, n in scen:
                d = fluid(segs, caps, n)[-1]
                dist.append((gp * sp, d))
        noise = self.p["noise"]
        full = []
        for pr, d in dist:
            for f, w in ((1 - 1.4 * noise, 0.25), (1.0, 0.5), (1 + 1.4 * noise, 0.25)):
                full.append((pr * w, max(0.0, d * f)))
        max_meals = int(BUDGET // COST[qi] + 1e-9)
        cands = sorted(set(min(max_meals, int(d)) for _, d in full) | {min(max_meals, int(d) + 1) for _, d in full})
        best_m, best_v, best_sold = 0, 0.0, 0.0
        for m in cands:
            if m <= 0:
                continue
            es = sum(pr * min(m, d) for pr, d in full)
            v = price * es - COST[qi] * m
            if v > best_v:
                best_m, best_v, best_sold = m, v, es
        # long-term value of the reputation change at this spot
        rep_bonus = 0.0
        if best_m > 0 and best_sold >= 1:
            new_rep = max(0.0, min(100.0, rep + REP_STEP[qi]))
            h = min(days_left, self.p["rep_horizon"])
            rep_bonus = self.p["rep_w"] * (new_rep - rep) * 0.006 * 12.0 * 70.0 * h / max(self.p["rep_horizon"], 1)
        order = {"spot": spot, "price": round(price, 2), "quality": QUALITIES[qi], "meals": best_m}
        return best_v + rep_bonus, order, best_v
