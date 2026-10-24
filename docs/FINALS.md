# Finals runbook

Design: `docs/superpowers/specs/2026-10-24-finals-autonomous-bot-design.md`. Baseline: tag
`live-baseline-day29` on `main`. This branch changes nothing about live behavior: the adaptive
profile steers only when `state["mode"] == "final"` (or `--profile final` in practice).

## What runs

`python3 bot.py` is the same command for live and the final. Every state it calls
`Runner.apply_profile`: final markets switch `Brain.use_final(True)`, other modes switch back.
While on live days the policy still **observes** (shadow learning), so the calibrator, competition
and rule models arrive at 8:30 AM already trained on live data.

In the final profile the planner:
- searches all spots (no smallest-crowd cap, no hand price caps); prices are limited only by the
  game and by customer value at our reputation;
- multiplies uncapped demand by `Calibrator.factor` (learned from predicted vs actual by price
  level, sellouts as lower bounds);
- subtracts `Competition.scaled(...)` (learned per-spot shortfall, rival pressure, exploration)
  and `RuleAdapter.spot_cost` (riots, learned unexplained terms) from each spot's score;
- takes risk, price aggressiveness and location breadth from `MetaController` (bounded, 3 arms
  per knob; resets partially each market);
- keeps rival modeling (attendance, location mixture incl. follow-us, pricing mixture incl.
  follow, phantom rival), crew planning, vote analysis, surprise scenarios;
- uses a staged price search (thin scenarios first) so six or more spots fit the 2 s window.

New spots register themselves (`game.register_spots`) from `RuleAdapter.normalize` when
`rules.spots` grows. Meat/veggie, wrecked trucks and coupon awareness live in `RuleAdapter`.

## Switches and files
| What | How |
|---|---|
| Force baseline (kill switch) | `NIGHTSHIFT_PROFILE=baseline` or `data/profile.json` `{"profile":"baseline"}` or `--profile baseline` |
| Force final in practice | `python3 bot.py --practice --profile final --days 30 --rivals 7 --seed X` |
| Final checkpoint | `data/final-runtime-v1.json` (never the live `runtime-v1.json`) |
| Seed priors | `python3 tools/train_final.py --data-root DATA --output model/final_priors.json` |
| Offline comparison | `python3 tools/final_eval.py --seeds a b --rivals 3 7 --days 30 --policies baseline final` |

## Restart checklist before 8:30 AM
1. Merge or run from `finals`; run the tests (`python3 -m unittest discover -s tests -q`).
2. Restart `python3 bot.py` during a day phase, never during an open order window; one instance.
3. Confirm the terminal prints `Profile: FINAL (adaptive)` when the first final state arrives.
4. Crew: the saved crew setting seeds each market; the planner re-plans it every night.

## Known limits
- Offline simulations are approximations; they are stress signals, not the official engine.
- Calibrator, competition and controller constants are untuned priors until data accumulates.
- Walk-aware fitting allocates walk-ins by sales share, an approximation.
- The riot forecast format is undocumented; warnings are read leniently from event text.
