"""
weather.py - weather-probability model for Night Shift Food Truck Wars.

Standard library only, Python 3.8+.  Holds no credentials.

What it does
------------
1. Reads the nightly forecast weather probabilities (clear/cloudy/rain/cold)
   defensively, whatever shape the API uses, and normalises them.
2. Keeps a per-spot, per-weather demand multiplier r[spot][weather]:
       actual_customers ~= expected_customers * r[spot][realised_weather]
   These are learned online (Bayesian shrinkage in log space) from
   state.last_day, so the bot calibrates itself instead of trusting guesses.
3. Turns the forecast into a demand *mixture*: one scenario per weather
   state, each with its own spread.  Use the scenarios for inventory
   (newsvendor) and spot choice; they capture the real risk that rain wipes
   out the Oval, which a single averaged number hides.

Why residual multipliers start at 1.0
-------------------------------------
We do not know whether forecast.expected_customers already includes the
weather.  If it does, adding "rain hurts the Oval" on top double-counts.
So the learned ratios start neutral.  The docs' qualitative knowledge
(rain hurts the Oval; Union and Thompson hold up) only enters as a weak
prior when RAW_PRIOR is switched on.  Either way, data overrides it in a
few days of the relevant weather.

Typical use in bot.py
---------------------
    import weather
    W = weather.WeatherModel.load()          # once at start-up

    # each night, before deciding:
    W.observe_last_day(state)                # learn from yesterday
    probs = weather.forecast_probs(state)
    for spot in SPOTS:
        mult = W.expected_multiplier(spot, probs)
        scen = W.demand_scenarios(spot, expected_customers[spot], probs)
        q    = W.demand_quantile(spot, expected_customers[spot], probs, 0.6)
    W.save()                                  # optional, bounded file
"""

import json
import math
import os

SPOTS = ("oval", "thompson", "union", "stadium", "highst", "northres")
WEATHERS = ("clear", "cloudy", "rain", "cold")

# Fallback when the forecast has no usable weather data.
DEFAULT_PROBS = {"clear": 0.4, "cloudy": 0.3, "rain": 0.2, "cold": 0.1}

# Weak priors from the docs' notes, used only if RAW_PRIOR is True (i.e. you
# believe expected_customers does NOT already include weather).
RAW_PRIOR = False
DOC_PRIOR = {
    "oval":     {"rain": 0.70, "cold": 0.85},   # outdoors, rain hurts
    "stadium":  {"rain": 0.85, "cold": 0.90},
    "highst":   {"rain": 0.92},
    "northres": {"rain": 0.95},
    "union":    {"rain": 1.05},                 # holds up / absorbs refugees
    "thompson": {"rain": 1.05},
}

PRIOR_STRENGTH = 3.0    # pseudo-observations behind each prior ratio
BASE_CV = 0.18          # demand noise not explained by weather
MAX_RATIO = 3.0         # clamp for bad or zero data
STATE_FILE = "weather_model.json"

_ALIASES = {
    "sunny": "clear", "sun": "clear", "fair": "clear",
    "overcast": "cloudy", "clouds": "cloudy", "cloud": "cloudy",
    "rainy": "rain", "storm": "rain", "showers": "rain", "wet": "rain",
    "snow": "cold", "freezing": "cold", "chilly": "cold",
}


def _canon(name):
    if not isinstance(name, str):
        return None
    k = name.strip().lower()
    k = _ALIASES.get(k, k)
    return k if k in WEATHERS else None


def _num(x, default=None):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def parse_probs(raw):
    """Normalise any plausible weather-probability shape to {weather: p}.

    Accepts {"rain": 0.3, ...}, {"rain": 30, ...} (percent),
    [{"type": "rain", "p": 0.3}, ...], or a plain string ("rain").
    Never raises; falls back to DEFAULT_PROBS.
    """
    out = {w: 0.0 for w in WEATHERS}
    try:
        if isinstance(raw, str):
            w = _canon(raw)
            if w:
                out[w] = 1.0
        elif isinstance(raw, dict):
            for k, v in raw.items():
                w = _canon(k)
                p = _num(v)
                if w and p is not None and p > 0:
                    out[w] += p
        elif isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                w = _canon(item.get("type") or item.get("weather")
                           or item.get("name") or item.get("kind"))
                p = _num(item.get("p", item.get("prob",
                         item.get("probability", item.get("chance")))))
                if w and p is not None and p > 0:
                    out[w] += p
    except Exception:
        pass
    total = sum(out.values())
    if total <= 0:
        return dict(DEFAULT_PROBS)
    if total > 1.5:                       # looks like percentages
        out = {w: p / 100.0 for w, p in out.items()}
        total = sum(out.values())
    return {w: p / total for w, p in out.items()}


def forecast_probs(state):
    """Pull weather probabilities out of state.forecast (any key name)."""
    fc = (state or {}).get("forecast") or {}
    if not isinstance(fc, dict):
        return dict(DEFAULT_PROBS)
    for key in ("weather", "weather_probs", "weather_probabilities",
                "weather_forecast", "probabilities"):
        if key in fc:
            return parse_probs(fc[key])
    # maybe the probabilities sit directly on the forecast
    if any(_canon(k) for k in fc):
        return parse_probs({k: v for k, v in fc.items() if _canon(k)})
    return dict(DEFAULT_PROBS)


def realised_weather(last_day):
    """Weather that actually happened yesterday, or None."""
    if not isinstance(last_day, dict):
        return None
    for key in ("weather", "actual_weather", "conditions"):
        v = last_day.get(key)
        if isinstance(v, dict):
            v = v.get("type") or v.get("name") or v.get("weather")
        w = _canon(v)
        if w:
            return w
    return None


def _spot_customers(info):
    """Total customers who showed up at a spot (uncensored by our stock)."""
    if not isinstance(info, dict):
        return _num(info)
    for key in ("customers", "arrivals", "visitors", "foot_traffic",
                "total_customers", "demand"):
        v = _num(info.get(key))
        if v is not None:
            return v
    return None


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _lognorm_params(mean, cv):
    s2 = math.log(1.0 + cv * cv)
    return math.log(max(mean, 1e-9)) - s2 / 2.0, math.sqrt(s2)


class WeatherModel(object):
    def __init__(self):
        # sufficient statistics per spot/weather, in log space
        self.sum_log = {s: {w: 0.0 for w in WEATHERS} for s in SPOTS}
        self.n = {s: {w: 0.0 for w in WEATHERS} for s in SPOTS}
        self.sum_sq = {s: {w: 0.0 for w in WEATHERS} for s in SPOTS}
        self._pending = None   # yesterday's forecast, to learn against

    # ----- learning -----------------------------------------------------
    def _prior(self, spot, w):
        if RAW_PRIOR:
            return DOC_PRIOR.get(spot, {}).get(w, 1.0)
        return 1.0

    def ratio(self, spot, w):
        """Posterior multiplier r[spot][w] (shrunk toward the prior)."""
        prior = math.log(self._prior(spot, w))
        n = self.n.get(spot, {}).get(w, 0.0)
        s = self.sum_log.get(spot, {}).get(w, 0.0)
        post = (PRIOR_STRENGTH * prior + s) / (PRIOR_STRENGTH + n)
        return min(MAX_RATIO, max(1.0 / MAX_RATIO, math.exp(post)))

    def residual_cv(self, spot, w):
        """Spread of demand around the weather-adjusted mean."""
        n = self.n.get(spot, {}).get(w, 0.0)
        if n < 3:
            return BASE_CV
        mean = self.sum_log[spot][w] / n
        var = max(0.0, self.sum_sq[spot][w] / n - mean * mean)
        # blend sample spread with the base, weighted by data
        sd = (3 * BASE_CV + n * math.sqrt(var)) / (3 + n)
        return min(0.8, max(0.05, sd))

    def update(self, spot, w, expected, actual):
        """Record one day: forecast said `expected`, `actual` showed up."""
        e, a = _num(expected), _num(actual)
        if spot not in self.n or w not in WEATHERS:
            return
        if e is None or a is None or e <= 0 or a < 0:
            return
        x = math.log(min(MAX_RATIO, max(1.0 / MAX_RATIO, (a + 0.5) / (e + 0.5))))
        self.n[spot][w] += 1.0
        self.sum_log[spot][w] += x
        self.sum_sq[spot][w] += x * x

    def remember_forecast(self, state):
        """Store tonight's per-spot expected customers to compare tomorrow."""
        fc = (state or {}).get("forecast") or {}
        exp = fc.get("expected_customers") if isinstance(fc, dict) else None
        if isinstance(exp, dict):
            self._pending = {s: _num(v) for s, v in exp.items() if s in SPOTS}

    def observe_last_day(self, state, expected_by_spot=None):
        """Learn from state.last_day.  Safe to call every night.

        expected_by_spot: the forecast you saw for that day; defaults to
        whatever remember_forecast() stored the night before.
        """
        try:
            ld = (state or {}).get("last_day") or {}
            w = realised_weather(ld)
            spots = ld.get("spots") if isinstance(ld, dict) else None
            exp = expected_by_spot or self._pending
            if w and isinstance(spots, dict) and exp:
                for s, info in spots.items():
                    if s in SPOTS and exp.get(s):
                        self.update(s, w, exp[s], _spot_customers(info))
        except Exception:
            pass
        self.remember_forecast(state)

    # ----- prediction ---------------------------------------------------
    def expected_multiplier(self, spot, probs):
        """E[r] under tomorrow's weather probabilities."""
        return sum(p * self.ratio(spot, w) for w, p in probs.items())

    def demand_scenarios(self, spot, expected, probs, points=5):
        """List of (probability, customers) scenarios for this spot.

        One block of `points` scenarios per weather state with p > 0, so the
        result reflects both weather risk and ordinary noise.  Feed it into
        your newsvendor / expected-profit loop.
        """
        e = max(0.0, _num(expected, 0.0) or 0.0)
        out = []
        # quantile midpoints of a standard normal, equally weighted
        zs = [_inv_norm((i + 0.5) / points) for i in range(points)]
        for w, p in probs.items():
            if p <= 0:
                continue
            mean = e * self.ratio(spot, w)
            mu, sig = _lognorm_params(mean, self.residual_cv(spot, w))
            for z in zs:
                out.append((p / points, math.exp(mu + sig * z)))
        return out or [(1.0, e)]

    def demand_cdf(self, spot, expected, probs, x):
        e = max(1e-9, _num(expected, 0.0) or 0.0)
        if x <= 0:
            return 0.0
        total = 0.0
        for w, p in probs.items():
            if p <= 0:
                continue
            mu, sig = _lognorm_params(e * self.ratio(spot, w),
                                      self.residual_cv(spot, w))
            total += p * _norm_cdf((math.log(x) - mu) / sig)
        return total

    def demand_quantile(self, spot, expected, probs, q):
        """Mixture quantile: e.g. the newsvendor critical-ratio demand."""
        e = _num(expected, 0.0) or 0.0
        if e <= 0:
            return 0.0
        q = min(0.999, max(0.001, q))
        lo, hi = 0.0, e * MAX_RATIO * 3 + 10
        for _ in range(50):
            mid = (lo + hi) / 2.0
            if self.demand_cdf(spot, e, probs, mid) < q:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    # ----- persistence (no secrets; bounded small file) ------------------
    def to_dict(self):
        return {"n": self.n, "sum_log": self.sum_log, "sum_sq": self.sum_sq}

    @classmethod
    def load(cls, path=STATE_FILE):
        m = cls()
        try:
            with open(path) as f:
                d = json.load(f)
            for attr in ("n", "sum_log", "sum_sq"):
                for s in SPOTS:
                    for w in WEATHERS:
                        v = _num(d.get(attr, {}).get(s, {}).get(w))
                        if v is not None:
                            getattr(m, attr)[s][w] = v
        except Exception:
            pass
        return m

    def save(self, path=STATE_FILE):
        try:
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.to_dict(), f)
            os.replace(tmp, path)
        except Exception:
            pass


def _inv_norm(p):
    """Inverse standard normal CDF (Acklam's approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - pl:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def adjust(model, spot, mean, sd, probs):
    """Weather-adjust a demand estimate you already have.

    Takes your planner's (mean, sd) for a spot and returns the (mean, sd) of
    the weather mixture: the mean moves by the expected weather multiplier,
    and sd grows when the forecast is uncertain (e.g. 50% rain at the Oval).
    Drop-in for code that uses a DEMAND_SD-style newsvendor.
    """
    mean = max(0.0, _num(mean, 0.0) or 0.0)
    sd = max(0.0, _num(sd, 0.0) or 0.0)
    m1 = m2 = 0.0
    for w, p in probs.items():
        if p <= 0:
            continue
        r = model.ratio(spot, w)
        mw, sw = mean * r, sd * r
        m1 += p * mw
        m2 += p * (sw * sw + mw * mw)
    if m1 <= 0:
        return mean, sd
    return m1, math.sqrt(max(0.0, m2 - m1 * m1))


def best_stock(model, spot, expected, probs, price, cost, budget=500.0,
               market_share=1.0):
    """Expected-profit-maximising meal count under the weather mixture.

    market_share: fraction of the spot's customers you expect to win
    (from your competitor model).  Returns (meals, expected_profit).
    """
    scen = [(p, d * market_share)
            for p, d in model.demand_scenarios(spot, expected, probs, 9)]
    max_meals = int(budget // cost) if cost > 0 else 0
    best = (0, 0.0)
    for q in range(0, max_meals + 1):
        ep = sum(p * min(d, q) for p, d in scen) * price - q * cost
        if ep > best[1]:
            best = (q, ep)
    return best


if __name__ == "__main__":
    # Self-test: simulate a world where rain cuts Oval demand to 60% and
    # boosts the Union to 115%, then check the model learns it.
    import random
    rng = random.Random(7)
    truth = {"oval": {"rain": 0.6, "cold": 0.8}, "union": {"rain": 1.15}}
    m = WeatherModel()
    for day in range(60):
        probs = parse_probs({"clear": 40, "cloudy": 30, "rain": 20, "cold": 10})
        w = rng.choices(WEATHERS, [probs[x] for x in WEATHERS])[0]
        exp = {s: 100.0 for s in SPOTS}
        actual = {s: {"customers": 100.0 * truth.get(s, {}).get(w, 1.0)
                      * math.exp(rng.gauss(0, 0.15))} for s in SPOTS}
        m._pending = exp
        m.observe_last_day({"last_day": {"weather": w, "spots": actual},
                            "forecast": {"expected_customers": exp}})
    for s in ("oval", "union", "thompson"):
        print(s, {w: round(m.ratio(s, w), 2) for w in WEATHERS})
    rainy = parse_probs({"rain": 0.8, "clear": 0.2})
    print("oval mult if 80% rain:", round(m.expected_multiplier("oval", rainy), 2))
    print("oval 60th pct demand (exp 100):",
          round(m.demand_quantile("oval", 100, rainy, 0.6), 1))
    print("best stock oval @ $9 premium:",
          best_stock(m, "oval", 100, rainy, price=9, cost=5.5))
    print("parse list shape:", parse_probs([{"type": "Rainy", "p": 0.5},
                                            {"type": "sunny", "p": 0.5}]))
    print("parse garbage:", parse_probs(None))
