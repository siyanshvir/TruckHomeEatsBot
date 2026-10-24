"""Benchmark bot versions on fixed OFFICIAL practice seeds. Run on your laptop.

    python3 benchmark.py                          # bot.py vs backup/bot_starter.py
    python3 benchmark.py bot.py                   # just one version

Uses the key from NIGHTSHIFT_KEY / .env via bot.py itself; never prints it.
Runs one practice market at a time with a pause between (rate-limit friendly).
Results go to benchmark_results.json (profits only, no credentials).
"""
import json
import re
import statistics
import subprocess
import sys
import time

SEEDS = ["11", "22", "33", "44", "55"]
RIVALS = [3, 6]
PAUSE = 2.0


def run(path, seed, rivals):
    cmd = [sys.executable, path, "--practice", "--seed", seed, "--rivals", str(rivals)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600).stdout
    except subprocess.TimeoutExpired:
        return None
    m = re.search(r"practice over: cash \$(-?[\d.]+)", out)
    return float(m.group(1)) if m else None


def summary(label, rows):
    p = [r["profit"] for r in rows if r["profit"] is not None]
    if not p:
        print("%s: no results" % label)
        return
    print("%-28s n=%d mean $%.0f median $%.0f sd $%.0f worst $%.0f" % (
        label, len(p), statistics.mean(p), statistics.median(p),
        statistics.pstdev(p), min(p)))
    for c in RIVALS:
        pc = [r["profit"] for r in rows if r["rivals"] == c and r["profit"] is not None]
        if pc:
            print("    rivals=%d mean $%.0f" % (c, statistics.mean(pc)))


if __name__ == "__main__":
    paths = sys.argv[1:] or ["backup/bot_starter.py", "bot.py"]
    results = {}
    for path in paths:
        rows = []
        for c in RIVALS:
            for s in SEEDS:
                prof = run(path, s, c)
                rows.append({"seed": s, "rivals": c, "profit": prof})
                print("  %s seed=%s rivals=%d -> %s" % (path, s, c, prof))
                time.sleep(PAUSE)
        results[path] = rows
    print()
    for path in paths:
        summary(path, results[path])
    if len(paths) == 2:
        a, b = results[paths[0]], results[paths[1]]
        d = [y["profit"] - x["profit"] for x, y in zip(a, b)
             if x["profit"] is not None and y["profit"] is not None]
        if len(d) > 1:
            print("paired gain %s over %s: mean $%.0f (se $%.0f)" % (
                paths[1], paths[0], statistics.mean(d), statistics.pstdev(d) / len(d) ** .5))
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=1)
