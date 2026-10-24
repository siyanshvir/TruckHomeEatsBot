"""Fit per-spot customer population parameters to collected practice data.

    python3 tools/fit.py data/collect.jsonl
"""
import json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bot as model

SPOTS = ["oval", "thompson", "union", "stadium", "highst", "northres"]


def load(path):
    obs = {s: [] for s in SPOTS}
    for line in open(path):
        r = json.loads(line)
        ld = r["last_day"]
        if not ld:
            continue
        reps = {t["id"]: t["rep"] for t in r["pre"]["trucks"]}
        for s in SPOTS:
            info = ld["spots"][s]
            if not info["trucks"]:
                continue
            offers = [dict(price=t["price"], quality=t["quality"], meals=t["meals"], rep=reps[t["id"]][s]) for t in info["trucks"]]
            obs[s].append((info["customers"], offers, [t["sold"] for t in info["trucks"]], ld["weather"]))
    return obs


def loss(params, data):
    mu, sd, pc, pp = params
    if sd < 0.3 or pc < 0 or pp < 0 or pc + pp > 1:
        return 1e18
    cells = model.population(mu, sd, pc, pp)
    tot = 0.0
    for count, offers, sold, _ in data:
        pred = model.expected_sold(count, cells, offers)
        for a, b in zip(pred, sold):
            tot += (a - b) ** 2
    return tot


def fit(data, seed=0):
    rng = random.Random(seed)
    best = [12.0, 2.0, 0.35, 0.2]
    bl = loss(best, data)
    step = [1.0, 0.6, 0.12, 0.08]
    for it in range(60):
        improved = False
        for i in range(4):
            for d in (+step[i], -step[i]):
                cand = list(best); cand[i] += d
                l = loss(cand, data)
                if l < bl:
                    best, bl, improved = cand, l, True
        if not improved:
            step = [s * 0.6 for s in step]
            if step[0] < 0.03:
                break
    return best, bl


if __name__ == "__main__":
    obs = load(sys.argv[1])
    out = {}
    for s in SPOTS:
        data = obs[s]
        if not data:
            continue
        p, l = fit(data)
        n = sum(len(d[2]) for d in data)
        base = sum(x ** 2 for d in data for x in d[2])
        print("%-9s n=%4d  mu=%.2f sd=%.2f cheap=%.2f prem=%.2f  rmse=%.1f  (null rmse %.1f)" % (s, n, p[0], p[1], p[2], p[3], (l / n) ** .5, (base / n) ** .5))
        out[s] = p
    json.dump(out, open("data/fit.json", "w"), indent=1)
