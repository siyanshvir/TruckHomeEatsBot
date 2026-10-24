"""Paired comparison in the offline simulator: v2 vs current bot.py vs starter.

    python3 tools/compare.py [seed_start] [n_seeds] [jitter]
"""
import json, os, statistics, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, bot
if os.environ.get("TRUTH") == "v1":
    sim.TRUTH_POP = dict(bot.POP)

a = int(sys.argv[1]) if len(sys.argv) > 1 else 50
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
j = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
seeds, rivals = list(range(a, a + n)), [3, 5, 8]
T = {"max": 0.0}


def v1(st, i, mk):
    st = dict(st); st["day"] = st["order_for_day"] - 1
    return bot.decide(st)


def timed(fn):
    def f(st, i, mk):
        t = time.time(); o = fn(st, i, mk); T["max"] = max(T["max"], time.time() - t); return o
    return f


out = {}
for name, fn in [("starter", sim.starter), ("bot.py (v1)", v1), ("bot_v2", timed(sim.make_strategy_bot()))]:
    r = sim.run_suite(fn, seeds, rivals, jitter=j)
    out[name] = {"%d-%d" % (s, k): round(c, 2) for s, k, c, _ in r}
    sim.summarize(name, r)
    print("    waste %.1f%%" % (100 * (1 - sum(x[3]["sold"] for x in r) / max(1, sum(x[3]["meals"] for x in r)))))
print("v2 max decision time: %.0f ms" % (T["max"] * 1000))
diff = [out["bot_v2"][k] - out["bot.py (v1)"][k] for k in out["bot_v2"]]
print("paired v2 - v1: mean %+.0f, v2 wins %d/%d" % (statistics.mean(diff), sum(d > 0 for d in diff), len(diff)))
os.makedirs("data", exist_ok=True)
json.dump(out, open("data/sim_compare_%d_%d_%s_%s.json" % (a, n, j, os.environ.get("TRUTH", "v2")), "w"), indent=1)
