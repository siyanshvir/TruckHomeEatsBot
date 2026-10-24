"""Real practice-API benchmark: same seeds and rival counts for every bot (paired comparison).

    python3 tools/benchmark.py bot bot_v2            # default seeds 101-105, rivals 3,5,8
    python3 tools/benchmark.py bot_v2 --seeds 201,202 --rivals 4

Plays practice markets only (never the live game). Reads the key from NIGHTSHIFT_KEY / .env
via the bot module; never prints it. Results -> data/bench_<bot>.json (data/ is gitignored).
"""
import argparse, importlib, json, os, statistics, sys, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)


def req(mod, method, path, body=None):
    for i in range(6):
        try:
            r = mod.call(method, path, body)
        except Exception:
            time.sleep(1 + i); continue
        status, js = r[0], r[1]
        if status == 429:
            time.sleep(min(float(r[2]) if len(r) > 2 and r[2] else 2.0 * (i + 1), 30)); continue
        return status, js
    return 0, {}


def one(mod, seed, rivals):
    s, r = req(mod, "POST", "/api/v1/practice", {"seed": str(seed), "days": 30, "rivals": rivals})
    if s not in (200, 201):
        return {"error": "start failed %s" % s}
    base = r["base"]
    m = dict(profit=0.0, revenue=0.0, cost=0.0, sold=0, meals=0, price=0.0, days=0, spots={}, quality={}, rep=[], hungry=0)
    t_max = 0.0
    for _ in range(200):
        s, st = req(mod, "GET", base + "/state")
        if s != 200:
            time.sleep(1); continue
        ld = (st.get("last_day") or {}).get("you") or {}
        if st.get("phase") == "over":
            m["profit"] = st["you"]["cash"]
            break
        if st.get("phase") != "night":
            time.sleep(0.5); continue
        t = time.time(); order = mod.decide(st); t_max = max(t_max, time.time() - t)
        s, res = req(mod, "POST", base + "/orders", order)
        if s != 200:
            return {"error": "order rejected %s %s" % (s, res.get("error"))}
        s, st2 = req(mod, "GET", base + "/state")
        y = (st2.get("last_day") or {}).get("you") or {}
        if y:
            m["revenue"] += y.get("revenue") or 0; m["cost"] += y.get("cost") or 0
            m["sold"] += y.get("sold") or 0; m["meals"] += y.get("meals") or 0
            m["price"] += y.get("price") or 0; m["days"] += 1
            m["spots"][y.get("spot")] = m["spots"].get(y.get("spot"), 0) + 1
            m["quality"][y.get("quality")] = m["quality"].get(y.get("quality"), 0) + 1
            rep = (st2.get("you") or {}).get("reputation") or {}
            if y.get("spot") in rep:
                m["rep"].append(rep[y["spot"]])
            sp = ((st2.get("last_day") or {}).get("spots") or {}).get(y.get("spot")) or {}
            m["hungry"] += sp.get("hungry") or 0
        if st2.get("phase") == "over":
            m["profit"] = st2["you"]["cash"]; break
    m["avg_price"] = m["price"] / max(1, m["days"]); m["waste"] = m["meals"] - m["sold"]
    m["avg_rep_at_spot"] = statistics.mean(m["rep"]) if m["rep"] else None
    m["max_decision_ms"] = round(t_max * 1000, 1)
    m.pop("rep"); m.pop("price")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bots", nargs="+")
    ap.add_argument("--seeds", default="101,102,103,104,105")
    ap.add_argument("--rivals", default="3,5,8")
    a = ap.parse_args()
    seeds = [x for x in a.seeds.split(",") if x]; rivals = [int(x) for x in a.rivals.split(",")]
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    allres = {}
    for b in a.bots:
        mod = importlib.import_module(b)
        res = {}
        for sd in seeds:
            for k in rivals:
                r = one(mod, sd, k); res["%s-%d" % (sd, k)] = r
                print("%-8s seed %-4s rivals %d: %s" % (b, sd, k, "$%.0f profit, sold %s/%s" % (r["profit"], r["sold"], r["meals"]) if "profit" in r else r))
        json.dump(res, open(os.path.join(ROOT, "data", "bench_%s.json" % b), "w"), indent=1)
        allres[b] = res
    print()
    for b, res in allres.items():
        p = [r["profit"] for r in res.values() if "profit" in r]
        if not p:
            continue
        print("%-8s mean %7.0f median %7.0f sd %6.0f worst %7.0f  n=%d" % (b, statistics.mean(p), statistics.median(p), statistics.pstdev(p), min(p), len(p)))
        for k in rivals:
            q = [r["profit"] for kk, r in res.items() if kk.endswith("-%d" % k) and "profit" in r]
            if q:
                print("    rivals %d: mean %.0f" % (k, statistics.mean(q)))
    if len(allres) >= 2:
        b0, b1 = list(allres)[:2]
        d = [allres[b1][k]["profit"] - allres[b0][k]["profit"] for k in allres[b0] if "profit" in allres[b0][k] and "profit" in allres[b1].get(k, {})]
        if d:
            print("paired %s - %s: mean %+.0f, wins %d/%d" % (b1, b0, statistics.mean(d), sum(x > 0 for x in d), len(d)))


if __name__ == "__main__":
    main()
