"""Contrarian strategy: park where the other trucks don't, and charge a little more.

The busy spots draw most of the trucks, so customers there get split many ways and prices get
undercut. The "least desirable" spot usually has more customers per truck. ContrarianBrain:
  1. predicts where rivals park (same sampling as Brain);
  2. picks the spot with the most expected customers per truck (us included);
  3. finds the best quality and price there with the regular demand model, then adds an upcharge;
  4. orders only as many meals as it expects to sell even on a slow day, so it still sells out.
If the contrarian order isn't expected to make money it falls back to the regular Brain.

    python3 bot.py --contrarian               # live game
    python3 bot.py --contrarian --practice    # practice market
"""
from strategy import BUDGET, COST, INF, QUALITIES, SPOTS, VALUE, Brain, _num, event_kinds, \
    fluid, build_segments, rep_factor, weather_probs

CONTRARIAN = {
    "upcharge": 0.50,       # dollars added on top of the model's best price at the quiet spot
    "upcharge_pct": 0.0,    # ...plus this share of that price
    "sellout": 0.25,        # order the meals we'd sell on a day this slow (quantile of demand)
    "min_crowd": 0.6,       # skip spots expected to draw under this share of the average spot's crowd
}


class ContrarianBrain(Brain):
    def __init__(self, params=None, seed=None):
        Brain.__init__(self, params, seed)
        for k, v in CONTRARIAN.items():
            self.p.setdefault(k, v)

    def decide(self, state):
        self.sync(state)
        p = self.p
        fc = state.get("forecast") or {}
        kinds = event_kinds(fc.get("events"))
        probs = weather_probs(fc)
        day = state.get("order_for_day") or 0
        days_left = max(0, int(state.get("days") or 30) - int(day))
        expected = self.forecast_customers(state, kinds)
        self.pending_forecast = (day, dict(expected), dict(probs))
        samples = self.rival_predictions(state, expected, day)
        my_reps = (state.get("you") or {}).get("reputation") or {}

        spot = self.quiet_spot(expected, samples)
        rep = _num(my_reps.get(spot, 50), 50)
        rf = rep_factor(rep)
        mu, sd, qmix = self.spot_model(spot, kinds)
        scen = self.scenarios(spot, expected[spot], probs)
        groups = self.spot_groups(samples, spot)

        # best quality and price at the quiet spot (no undercut ceiling: we're here to charge more)
        base = None
        for qi in range(3):
            for price in self._price_grid(qi, rf, groups):
                res = self._evaluate(spot, qi, price, rf, rep, mu, sd, qmix, scen, groups, days_left)
                if base is None or res[0] > base[0]:
                    base = res
        best = None
        if base is not None:
            qi = QUALITIES.index(base[1]["quality"])
            price = base[1]["price"] * (1 + p["upcharge_pct"]) + p["upcharge"]
            price = round(min(price, 30.0, VALUE[qi] * rf - 0.01), 2)
            dist = self._demand(qi, price, rf, mu, sd, qmix, scen, groups)
            meals = min(int(BUDGET // COST[qi]), int(self._quantile(dist, p["sellout"])))
            sold = sum(pr * min(meals, d) for pr, d in dist)
            profit = price * sold - COST[qi] * meals
            if meals > 0 and profit >= p["min_profit"]:
                best = {"spot": spot, "price": price, "quality": QUALITIES[qi], "meals": meals}
        watch = self.rival_watch(samples)
        if best is None:
            order = Brain.decide(self, state)       # quiet spot doesn't pay today: play it normally
            self.rival_note = "contrarian: %s doesn't pay today, regular brain instead" % spot
            return order
        self.rival_note = "contrarian: %s ($%.2f, %d meals, ~$%.0f) | %s" % (
            spot, best["price"], best["meals"], profit, self._rival_note(watch, spot))
        self.days_at[spot] += 1
        return best

    def quiet_spot(self, expected, samples):
        """The spot with the most expected customers per truck, ignoring spots that are nearly empty."""
        n = max(1, len(samples))
        avg = sum(expected.values()) / max(1, len(expected))
        best, best_score = None, -1.0
        for s in SPOTS:
            if expected[s] < self.p["min_crowd"] * avg:
                continue
            trucks = sum(len(smp[s]) for smp in samples) / n
            score = expected[s] / (trucks + 1.0)
            if score > best_score:
                best, best_score = s, score
        return best or max(expected, key=expected.get)

    def _demand(self, qi, price, rf, mu, sd, qmix, scen, groups):
        """Our demand distribution [(prob, meals)] at this price, as in Brain._evaluate."""
        dist = []
        for gp, rivals in groups:
            trucks = [(r[1], r[2], r[3]) for r in rivals] + [(price, qi, rf)]
            caps = [r[4] for r in rivals] + [INF]
            segs = build_segments(trucks, mu, sd, qmix)
            for sp, n in scen:
                d = fluid(segs, caps, n)[-1]
                for f, w in ((1 - 1.4 * self.p["noise"], 0.25), (1.0, 0.5), (1 + 1.4 * self.p["noise"], 0.25)):
                    dist.append((gp * sp * w, max(0.0, d * f)))
        return dist

    @staticmethod
    def _quantile(dist, q):
        acc = 0.0
        for pr, d in sorted(dist, key=lambda x: x[1]):
            acc += pr
            if acc >= q:
                return d
        return max((d for _, d in dist), default=0.0)
