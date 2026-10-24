# R3 experiments: knob sweep of the adaptive final profile

Script: `tools/research_sweep.py` (run with `nice -n 15`, 2 workers). Raw per-market results:
`data/research-sweep.jsonl` (git-ignored; one line per config x market). Regenerate tables with
`python3 tools/research_sweep.py report --split`.

## Setup
- `final_mode.make_final_brain(samples=16)` fresh per market; `controller` replaced by a stub that always
  returns fixed `Knobs(price_aggr, breadth, risk)` (so the learned bandit is OFF in every config; the default
  config is "fixed 1.0/0.5/0.10", not the live adaptive controller). Calibrator/competition set to None for E4/E5.
- `sim.Market(seed, rivals, 20 days, style, features fair_choice+walk_over+salvage)`, seeds r3a-r3c,
  rivals {3,7}, styles {mixed, brutal, moving, follower}: 24 planned markets per config. Time budget stopped the
  run early: **15 fully paired markets** (14 for baseline, unrestricted, nocomp).
  Styles are unevenly filled (follower n=3, others n=4). Sweep order was market-by-market so coverage is balanced.
- Profit = final cash after 20 days. SE = std of paired differences / sqrt(n). Sim is approximate, reputation
  resets per market, no cross-market learning is exercised.

## Results (20 days, paired vs fixed-default final)
| config | n | mean profit | std | worst market | worst day | waste | s/decision | paired diff | SE | W/L |
|---|---|---|---|---|---|---|---|---|---|---|
| final (1.0/0.5/0.10) | 15 | 8095 | 2025 | 4872 | -354 | 386 | 1.03 | - | - | - |
| risk 0.03 | 15 | 8062 | 2019 | 4872 | -354 | 395 | 1.04 | -33 | 60 | 3/4 |
| risk 0.30 | 15 | 8069 | 2063 | 4057 | -354 | 376 | 1.04 | -26 | 184 | 7/7 |
| price 0.9 | 15 | 7939 | 2044 | 3773 | -349 | 371 | 1.04 | -156 | 211 | 6/9 |
| price 1.1 | 15 | 8097 | 2105 | 4356 | -360 | 410 | 1.05 | +2 | 93 | 6/9 |
| breadth 0.0 | 15 | 8077 | 1971 | 4872 | -354 | 395 | 1.07 | -18 | 123 | 3/7 |
| breadth 1.0 | 15 | 7842 | 2034 | 4473 | -354 | 379 | 1.06 | -253 | 117 | 3/9 |
| calibrator off | 15 | 7658 | 2378 | 3211 | -374 | 620 | 1.05 | -437 | 229 | 4/11 |
| competition off | 14 | 8080 | 2048 | 4872 | -354 | 397 | 1.02 | -67 | 144 | 3/8 |
| baseline | 14 | 4988 | 1427 | 2596 | -278 | 319 | 0.67 | -3160 | 588 | 0/14 |
| unrestricted | 14 | 7385 | 3018 | 842 | -374 | 639 | 2.82 | -763 | 336 | 2/12 |

(Ties count in neither W nor L. Paired diffs for the last three use the 14 markets all share.)

## Reading the numbers
Supported (|diff| >= ~2 SE, n=14-15):
- **Final profile >> baseline**: +3160 (SE 588), 14/14 markets. Robust in every style.
- **Final >> unrestricted** (same search, no calibrator/competition/knob shaping): +763 (SE 336), 12/14, and
  unrestricted has a far worse worst market (842) and ~2.7x decision time. The learned shaping matters, not just uncapping.
- **Calibrator on** beats off: +437 (SE 229, ~1.9 SE), 11/15 wins, waste 620 vs 386, worst market 3211 vs 4872.
  Moderate evidence; only "follower" style went the other way (+407 for off, n=3).
- **Breadth 1.0 hurts**: -253 (SE 117, ~2.2 SE), 9 of 12 decided markets; negative in 3 of 4 styles and both rival counts.

Noise (within ~1 SE; do not act on): risk 0.03 / 0.30, price 1.1, breadth 0.0, competition off (-67, SE 144),
price 0.9 (-156, SE 211, but sign negative in 3 of 4 styles). Style-level splits (n=3-4 each) are too small to trust;
e.g. risk 0.30 shows +217/+287 in mixed/moving and -393 in brutal, and breadth 0.0 +234 in moving, but these
are 1-SE-scale wiggles of a market-to-market std of ~2000.

## Recommended defaults
Keep the middle arm everywhere (price 1.0, breadth 0.5, risk 0.10): no alternative beat it beyond noise, and the
only significant knob result (breadth 1.0) is negative. Keep calibrator on; keep competition on (neutral to slightly
helpful in 3-rival markets, -176 vs +77 at 7 rivals is noise; costs nothing in time). Since the bandit explores
the outer arms, consider damping exploration of breadth 1.0 (e.g. lower its prior mean or drop that arm) and of
price 0.9; the outer arms for risk and price 1.1 are cheap (within noise). Do not run unrestricted without shaping.

## Limits
- 15 paired markets, 20 days: detectable differences are roughly >= 250-300 in profit (market std ~2000, but paired
  SE 60-230). Smaller effects (risk, price 1.1) are undetermined, not shown to be zero.
- Fixed knobs, not the live adaptive controller: this measures the arms, not the bandit's switching/learning
  gain, and not the controller's streak-based caution context.
- Approximate sim; reputation reset each market; seeds r3a-r3c only; no 30-day horizon; one run each (profit
  deterministic per seed, sampling noise inside the planner not replicated).
- Suggested next step: widen to 10 seeds for E3 (breadth) and E4 (calibrator) only, and add a controller-on arm.
