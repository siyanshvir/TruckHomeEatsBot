# Finals autonomous bot — design

Branch: `finals`. Baseline: tag `live-baseline-day29` on `main`. Live code is frozen; nothing here
changes live behavior. The final profile activates only when `state["mode"] == "final"` (and in
practice with `--profile final`).

## 1. Goal and success

The final (8:30 AM Eastern) is 20 seeded markets × 30 days, every truck in every market, ranked by
**average profit per market**. Each market starts at $0 cash and reputation 50 everywhere; rules live
at 8:30 AM apply from day 1; new spots open 4 of 8 on seeded days; polls close before days 8 and 19;
the crew setting on our truck seeds every market; each day waits up to 2 seconds for an order.

**The final bot must run with no human input.** Nothing may depend on someone editing a number
(price ceilings, location limits, risk, crew, votes) when conditions change. It learns from all data
available (live days to date, practice, and the final's own markets as they play) and adapts.

Success criteria (checked before promotion, §10):
- It never misses a day (every day answered in well under 2 s, errors never stop the loop).
- On matched practice/offline seeds it beats the frozen baseline on average profit **and** does not
  materially worsen tail losses or any individual market regression it should have avoided.
- Every adaptive piece degrades safely to the baseline behavior when it has no data.

Non-goals: reinforcement-learning frameworks, external dependencies (standard library only), any
change to live `main` unless the user explicitly says so.

## 2. Approach: self-tuning layer (A) + small meta-controller (B-lite)

Keep the proven planner (`strategy.Brain`: scenarios, learned customer/rival models, exhaustive
inventory search, frozen ledger). Replace every hand-set policy number with a value that is
*learned from results*, and add a small bandit over three coarse knobs. No whole-strategy switching.

```
state ─► RuleAdapter (normalize rules, new spots, food, riots, unknown terms)
      ─► Brain.scenarios (rivals incl. follow-us, phantom, attendance; crowds; surprises)
      ─► evaluate(spot, quality, price)  ── demand × Calibrator.factor()  ── profit (Overheads)
      ─► score = profit − risk·loss + reputation/exploration − Competition.penalty(spot)
      ─► knobs from MetaController (price_aggr, breadth, risk)  ── pick order
      ─► staff plan, vote analysis, coupon stance, food choice (all learned/model-based)
      ─► after the day: observe results ─► Calibrator, Competition, MetaController, rivals, crowds
```

## 3. Components and interfaces

Each module is standard-library Python, persists via `to_dict()/from dict`, tolerates missing data,
and never raises into the order path (the caller wraps with try/except and falls back to baseline).

### 3.1 `calibration.py` — learned price/quality optimism (owner: calibration agent)
Problem it solves: day 28 predicted 84 sales at $10.92 and sold 53; customer price-response above
the range we had sold at was a guess, and local preference fitting is paused under walk-over.
```
Calibrator(data=None); .to_dict(); .observe(check)  # check = a Brain.prediction_checks entry plus
    # keys: spot, quality, price (charged), value_ceiling (customer value at our reputation),
    #       weather, sellout (bool), expected_demand, actual_sales, meals
.factor(spot, quality, price, value_ceiling, weather=None) -> float in [0.3, 1.15]   # multiplies uncapped demand
.report() -> dict
```
Buckets: quality × price/value_ceiling ratio, with hierarchical shrinkage toward the global ratio.
A sellout is a lower bound (never pushes the factor down). A tail of bad predictions at high price
ratios must pull the factor down for similar prices within 2–3 observations. Starts at 1.0 with
modest uncertainty. Mean factor on data lacking high-price evidence stays near 1.

### 3.2 Walk-aware customer learning (owner: learning agent, `learning.py`)
Under `walk_over` the learner currently withholds local preference fitting. The API reports
`walked_in` and `walked_out` per spot. Restore fitting by removing walk-ins from local sales
(`local ≈ sold − walked_in`, allocated across trucks at the spot by sales share) and treating
`walked_out`/`hungry` consistently, so the price-response model keeps learning in the final.
Keep regime banks separate (`feature_tuple`), keep checkpoint compatibility, keep sellout censoring.

### 3.3 `competition.py` — dynamic location choice (owner: controller agent)
No hard "three smallest" cap in the final. All spots are searched. A learned per-spot adjustment
`Competition.penalty(spot, state) -> dollars/day` captures what the scenario model misses:
- per-spot EMA of (our realized profit − predicted profit) at that spot, shrunk to zero;
- recent rival pressure (trucks seen at the spot, followers of our truck, undercut depth);
- an exploration bonus for spots we have not tried in this market while reputation is still 50.
Interface: `.observe(last_day, our_order, predicted)`, `.penalty(spot, state) -> float`, `.to_dict()`.
The `breadth` knob scales how much weight this penalty and the exploration bonus receive.

### 3.4 `controller.py` — meta-controller (owner: controller agent)
Three coarse knobs, each an independent discounted Thompson-sampling bandit over 3 arms, rewarded
with realized daily profit relative to the model's own prediction (so it learns *which settings the
model over/under-trusts*, not noise):
- `price_aggr` ∈ {0.90, 1.00, 1.10}: scales the demand-side calibrated price headroom;
- `breadth` ∈ {0.0, 0.5, 0.75} (1.0 dropped: worse by about 2.2 se in the sweeps): weight on the competition penalty/exploration;
- `risk` ∈ {0.03, 0.10, 0.30}: expected-loss weight (final default prior centered near 0.10;
  the final ranks an average over many days, so near risk-neutral is correct in expectation, while
  recent losing streaks raise it temporarily).
Interface: `MetaController(data=None, seed=0)`, `.choose(context) -> Knobs`, `.update(knobs, reward)`,
`.to_dict()`. Priors come from `model/final_priors.json` (§6). Within a market the controller
exploits; at each market start it forgets partially (reputation-dependent regime). It may not move
outside its arm ranges — it is a bounded tuner, not a free policy.

### 3.5 Rival modeling (existing, owned by integration)
Already in `main`/baseline: attendance, mixture of location experts (uniform, habit, repeat, crowd,
context, opportunity, **follow** — follows our truck onto spots we just left), pricing experts
(context, repeat, undercut, cycle, **follow** — undercuts our last price by a learned depth), phantom
(unforeseen rival at a spot, learned rate), conditional rival profiles in the frozen ledger.
Final additions: rival state resets are *not* applied across markets (behavior is learned across
markets), but per-market reputations reset. Known rival styles (chaser, camper, undercutter,
copycat) in the offline simulator include followers so the evaluation exercises this.

### 3.6 `rules_adapter.py` — rules we know and rules to come (owner: rules agent)
Single place that turns raw state/features into planner inputs, so new rules never touch the planner:
- **New spots** (`rules.spots` grows): priors from the docs table (weight, price limit, premium
  share), walking distances from `spot_info` coordinates, move costs from `rules.move_costs`.
  The location search then includes them.
- **Meat or veggie** (`features.food`): choose `food` per order from `forecast.diets` and visible
  rival food mix; default `meat` when unknown.
- **Riots** (`features.riots`, `forecast` warnings): expected repair/wreck cost by spot from the
  documented formulas; treat 403 wrecked as non-fatal.
- **Coupons**: defensive awareness only (`you.coupons`); no offensive use unless a measured
  expected gain is positive (default off).
- **Planned counts**: used as evidence of crowding at each spot when present.
- **Unknown terms**: any new profit term the API reports in `last_day.you` that we do not model is
  learned as a residual by (spot, quality) and fed to the score; unknown features are ignored safely.
Interface: `RuleAdapter(data=None)`, `.normalize(state) -> state`, `.order_extras(state, order)`,
`.spot_cost(state, spot) -> dollars`, `.observe(last_day, our_order, modeled_profit)`, `.to_dict()`.

### 3.7 Staff, votes, protests (existing models, made self-contained)
- Crew: `staffing.CrewModel` (learned pool/poach rates, predicted-vs-actual checks). In the final it
  also chooses the starting crew; the saved truck setting seeds each market.
- Votes: `politics.vote_analysis` (win odds from the unvoted trucks, protest cost, margin). Law
  effects use API params and are also learned from observed realized `taxes/subsidies/price_charged`.
- Law crowd multipliers are validated empirically from day-one law data and learned if forecasts
  already include them (`Calibrator`/`Competition` absorb a persistent bias).

### 3.8 Integration (owner: lead)
`final_mode.py` holds `FinalPolicy` wiring the above into `Brain` through narrow hooks:
calibrated demand in `evaluate`, `Competition.penalty` in the score, knob-driven risk/breadth/price
headroom, rule adapter inputs, and the location restriction **off**. `bot.py` selects the profile by
`state["mode"]` at each state, handles market boundaries (day resets, new market key), and keeps a
kill switch (`NIGHTSHIFT_PROFILE=baseline` or `data/profile.json`) that forces baseline behavior.

## 4. Per-market and cross-market state
- Reset each market: reputations, vote/news state, crew hiring, market-local knob statistics.
- Carry across markets: customer/crowd learning, rival behavior models, calibrator, competition
  priors (partially discounted), controller arm statistics (discounted).
- One checkpoint file for the final (`data/final-runtime.json`), never the live checkpoint.

## 5. Safety and timing
- Every adaptive call is wrapped; any exception logs and falls back to the baseline value.
- Planning time budget < 0.9 s; the existing decision path is ~0.4 s. The order POST precedes all
  disk writes. Policy exceptions fall back to the baseline `fallback()` order.
- 403 (wrecked) and 409 are non-fatal; the loop keeps polling.
- One live process only; the final profile never writes the live checkpoint.

## 6. Training data
All data is chronological and leak-free. `dataset.py` normalizes live observations
(`data/live-observations`), the evaluation/forecast ledgers, practice logs, and final logs into day
records. `tools/train_final.py` fits calibration and controller priors into
`model/final_priors.json` (never overwriting `model/prior.json`). Final-market data is appended to
`data/final-*.jsonl` for later combined training with live data.

## 7. Evaluation
`tools/final_eval.py` runs matched seeds offline (`sim.py` with brutal, moving, **follower** and
undercutter rivals, new rules on/off, reputation reset) and on private practice (10 markets/hour
limit) comparing baseline vs final profile: average profit, tail loss, worst market, waste, zero-sale
days, latency. Report per-market regressions, not only averages.

## 8. Failure handling summary
| Situation | Behavior |
|---|---|
| Module raises | baseline value for that factor, logged |
| Missing/old checkpoint field | defaults/prior, never crash |
| New unknown rule/field | ignored by the planner; residual learned if it appears in profit terms |
| API 403/409/429 | no tight loops; re-poll; baseline order logic |
| Slow plan | smaller sample count; cached order reused |

## 9. Future hooks (deliberately left open)
New rules plug into `RuleAdapter`; new rival behaviors plug into `opponents` experts; new knobs
are added to `MetaController` as a bounded arm set; priors are regenerated by `train_final.py`.

## 10. Promotion gates (user decides on `main`)
1. All tests pass (including regression of baseline behavior when the profile is off).
2. Matched offline + practice evaluation: average profit up, tail loss not materially worse.
3. Latency < 1 s over all evaluation markets; 20 consecutive 30-day markets run without error.
4. Kill-switch verified. 5. User explicitly approves merging anything into `main`.
