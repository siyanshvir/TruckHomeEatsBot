"""Food Truck Wars starter bot. Python 3.8+, standard library only.

    export NIGHTSHIFT_KEY=fk_your_key_here
    python3 bot.py              # play the live game (leave it running)
    python3 bot.py --practice   # play a private practice market and print your profit

The strategy lives in strategy.py (see its docstring); sim.py is an offline simulator for tuning it.
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

from strategy import BUDGET, COST, QUALITIES, Brain


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


BRAIN = Brain()
USE_STARTER = False      # --starter: play the original starter strategy (for before/after comparisons)


def starter_decide(state):
    """The original starter strategy: random spot, standard at $10, a full budget of meals."""
    rules = state["rules"]
    return {"spot": random.choice(rules["spots"]), "price": 10.0, "quality": "standard",
            "meals": int(rules["budget"] // rules["costs"]["standard"])}


def safe_default(state):
    """A cautious order used only if the strategy fails: standard meals at our best-reputation spot."""
    try:
        reps = state["you"]["reputation"]
        spot = max(reps, key=lambda s: float(reps[s]))
    except Exception:
        spot = "union"
    return {"spot": spot, "price": 10.5, "quality": "standard", "meals": 55}


def decide(state):
    """Return tomorrow's order, or None to sit the day out. Never raises."""
    if USE_STARTER:
        return starter_decide(state)
    try:
        order = BRAIN.decide(state)
    except Exception as e:
        print("  strategy error (%r), using the safe default" % (e,))
        return safe_default(state)
    if order is None:
        return None
    try:
        rules = state.get("rules") or {}
        quality = order["quality"]
        cost = float((rules.get("costs") or {}).get(quality, COST[QUALITIES.index(quality)]))
        budget = float(rules.get("budget", BUDGET))
        order["meals"] = max(1, min(int(order["meals"]), int(budget // cost)))
        order["price"] = round(max(float(rules.get("price_min", 1)), min(float(rules.get("price_max", 30)),
                                                                       float(order["price"]))), 2)
        return order
    except Exception:
        return safe_default(state)


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
        except (urllib.error.URLError, OSError, ValueError) as e:
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
        if not isinstance(state, dict) or "version" not in state:
            time.sleep(1)
            continue
        version = state["version"]
        if state.get("phase") == "over" and state.get("mode") == "practice":
            report(state)
            return state
        key = (state.get("mode"), state.get("market"), state.get("order_for_day"))
        if state.get("phase") == "night" and state.get("orders_open") and key != ordered_for:
            try:
                report(state)
            except Exception:
                pass
            order = decide(state)
            if order is None and state.get("mode") == "practice":
                # a practice day only advances once we order: buy one meal nobody will pay for
                order = {"spot": "stadium", "price": 30.0, "quality": "premium", "meals": 1}
            if order is None:
                ordered_for = key
                print("  sitting out day %s" % state.get("order_for_day"))
                continue
            try:
                status, res = call("POST", base + "/orders", order, timeout=10)
            except (urllib.error.URLError, OSError, ValueError) as e:
                print("  network error sending the order (%s), retrying" % e)
                time.sleep(0.3)
                continue
            if status == 200 and isinstance(res, dict):
                ordered_for = key
                print("  ordered for day %s: %s" % (res.get("for_day"), order))
                if "state" in res:          # practice plays the day right away
                    version = 0
            elif status == 409:            # orders closed or wrong phase: wait for the next state
                ordered_for = key
                print("  order not accepted (409): %s" % res.get("error"))
            elif status == 429:
                time.sleep(1)
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
    ap.add_argument("--rivals", type=int, help="practice house rivals")
    ap.add_argument("--starter", action="store_true", help="play the original starter strategy instead")
    args = ap.parse_args()
    global USE_STARTER
    USE_STARTER = args.starter
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
