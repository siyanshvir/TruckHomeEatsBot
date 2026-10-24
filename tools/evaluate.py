"""Score bot.decide over seeded practice markets.

    python3 tools/evaluate.py "HORIZON=6,REP_WEIGHT=0.6" "HORIZON=0" ...
Each argument is one variant (comma-separated GLOBAL=value overrides of bot.py).
"""
import os, sys, threading, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bot

SEEDS = ["s1", "s2", "s3", "s4", "s5", "s6"]
RIVALS = [3, 7]


def call(*a, **k):
    for i in range(6):
        try:
            return bot._call(*a, **k)
        except Exception:
            time.sleep(1 + i)
    return 0, {}


bot._call = bot.call
bot.call = call


def game(seed, rivals):
    s, r = bot.call("POST", "/api/v1/practice", {"seed": seed, "days": 30, "rivals": rivals})
    base = r["base"]
    s, st = bot.call("GET", base + "/state")
    while st["phase"] == "night":
        s, res = bot.call("POST", base + "/orders", bot.decide(st))
        if s != 200:
            return None
        s, st = bot.call("GET", base + "/state")
    cash = {t["id"]: t["cash"] for t in st["trucks"]}
    me = st["you"]["id"]
    best_rival = max(v for k, v in cash.items() if k != me)
    return cash[me], best_rival


def run(label):
    jobs = [(s, r) for s in SEEDS for r in RIVALS]
    res = {}

    def w(chunk):
        for j in chunk:
            res[j] = game(*j)
    ts = [threading.Thread(target=w, args=(jobs[i::3],)) for i in range(3)]
    [t.start() for t in ts]; [t.join() for t in ts]
    mine = [v[0] for v in res.values() if v]
    wins = sum(1 for v in res.values() if v and v[0] > v[1])
    by = {r: sum(v[0] for (s, rr), v in res.items() if v and rr == r) / len(SEEDS) for r in RIVALS}
    print("%-40s avg %8.0f  min %8.0f  wins %d/%d  by rivals %s" % (label, sum(mine) / len(mine), min(mine), wins, len(mine), {k: round(v) for k, v in by.items()}), flush=True)


if __name__ == "__main__":
    for arg in sys.argv[1:] or [""]:
        for kv in filter(None, arg.split(",")):
            k, v = kv.split("=")
            setattr(bot, k, float(v) if "." in v else int(v))
        run(arg or "default")
        import importlib; importlib.reload(bot); bot._call = bot.call; bot.call = call
