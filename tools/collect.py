"""Collect (pre-state, last_day) pairs from practice markets played with random orders.

    python3 tools/collect.py GAMES THREADS   # writes data/collect.jsonl
"""
import json, os, random, sys, threading
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bot

lock = threading.Lock()
_call = bot.call


def call(*a, **k):
    for i in range(6):
        try:
            return _call(*a, **k)
        except Exception:
            import time; time.sleep(1 + i)
    return 0, {}


bot.call = call


def one_game(seed, out):
    rng = random.Random(seed)
    s, r = bot.call("POST", "/api/v1/practice", {"seed": str(seed), "days": 30, "rivals": rng.choice([3, 5, 7])})
    if s not in (200, 201):
        return
    base = r["base"]
    s, st = bot.call("GET", base + "/state")
    while st["phase"] == "night":
        pre = {"day": st["order_for_day"], "forecast": st["forecast"],
               "trucks": [{"id": t["id"], "rep": t["reputation"]} for t in st["trucks"]]}
        spot = rng.choice(st["rules"]["spots"])
        q = rng.choice(["cheap", "standard", "premium", "premium", "standard"])
        price = round(rng.uniform(5, 17), 1)
        meals = rng.choice([40, 80, 120, 160])
        meals = min(meals, int(500 // st["rules"]["costs"][q]))
        s, res = bot.call("POST", base + "/orders", {"spot": spot, "price": price, "quality": q, "meals": meals})
        if s != 200:
            return
        s, st = bot.call("GET", base + "/state")
        rec = {"seed": seed, "pre": pre, "last_day": st["last_day"], "me": st["you"]["id"]}
        with lock:
            out.write(json.dumps(rec) + "\n"); out.flush()


def worker(seeds, out):
    for sd in seeds:
        one_game(sd, out)


if __name__ == "__main__":
    games, threads = int(sys.argv[1]), int(sys.argv[2])
    out = open("data/collect.jsonl", "a")
    ts = [threading.Thread(target=worker, args=(list(range(1000 + i, 1000 + games, threads)), out)) for i in range(threads)]
    [t.start() for t in ts]; [t.join() for t in ts]
    print("done")
