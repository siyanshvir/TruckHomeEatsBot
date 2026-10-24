# Truck Home Eats — Night Shift bot

Python 3.8+, standard library only. Learns customer demand and competitor habits,
then optimizes quality, price, and inventory within your location and pricing rules.

Configure `NIGHTSHIFT_KEY` in your existing `.env` or environment. Keep it private.

```sh
python3 bot.py                                         # live game and final
python3 bot.py --practice --seed check-a --rivals 5      # private practice
python3 -m unittest discover -s tests -v
```

Start only one live process for your truck. A GitHub push does not restart an
already-running bot. This rebuild removes the old starter/contrarian/learning
entry points; use `bot.py`.

## Current location and pricing rules

Production orders are restricted to the **three smallest customer forecasts**.
The two smallest receive preference. The third can win only when its predicted
profit exceeds the best of the bottom two by at least $15 or 15%, whichever is
larger, and its risk-adjusted score is also better. The three busier spots are
excluded regardless of their predicted profit. Event/weather effects are already
included in the supplied forecast counts used for this ranking.

Price ceilings are **$5.50 cheap, $7 standard, $8.50 premium**. The optimizer can
charge less when supported by its sales forecast, with a $7.25 premium floor.
Production uses an additional downside-loss weight of 0.08 (previously 0.15);
all inventory still pays full ingredient costs, including unsold meals, and
quantities are optimized against forecast demand rather than automatically increased.
The fallback also uses the
bottom two and caps standard at $7. These are deliberate user constraints and
can reduce achievable profit compared with unrestricted optimization.

## Decisions

Each decision evaluates all six locations and all three qualities before any
price refinement. Crowd size alone never determines location. Each spot receives
equal search effort, without an early cutoff that can skip quieter spots.

The model learns a distribution over customer budget and minimum-quality
preferences from observed sales across the market. Sequential customer-choice
modeling includes reputation, affordability, stockouts, and overflow. A sellout
is a lower bound on demand, not evidence that demand stops at the sold count.
Rival scenarios use a learned mixture of location models: uniform uncertainty,
multi-day habits, repeat behavior, crowd chasing, context matching, and opportunity
seeking. Historical neighbors match tomorrow's announced events, weather odds,
and crowd shares. Pricing combines contextual historical prices with repeat,
undercut, and price-cycle hypotheses. Quality and stock use multiple historical
observations. Attendance is learned from both orders and absences. Only
registered rivals are modeled; no invented extra competitors are added.

For each location/quality/price, every feasible integer inventory, including
zero, competes on expected revenue minus food costs and a downside-loss penalty.
Production's contribution to campus demand is approximated at three levels and
interpolated. A bounded reputation benefit declines toward the end of a market;
a small information bonus encourages early exploration. Prices obey the low ceilings above, the game's limits, and the maximum
nonnegative customer value for that quality and reputation. Within those limits,
they do not have to undercut every rival: overflow and reputation can still matter.

`forecast.expected_customers` is the starting point. It already incorporates
weather/events. Weather scenarios are normalized around those counts, and online
calibration learns remaining forecast errors. Explicit zero forecasts stay zero.

## Learning and persistence

`model/prior.json` is trained from saved practice collection pairs. The initial
420 training days and 120 held-out days come from disjoint markets. The holdout
stays out of the shipped prior. Training and evaluation load no credentials.

```sh
python3 tools/train.py --data data/collect.jsonl
python3 tools/evaluate.py --seeds unseen-a unseen-b --rivals 3 7
python3 tools/evaluate.py --server --seeds check-a check-b --rivals 3 7
```

The server evaluator uses private practice only. Every policy/market has its own
learner; comparisons never share mutable learning state. `cautious` is a simple
premium-at-$9.25 baseline using forecast minus yesterday's rival stock;
`untrained` uses the optimizer without the saved prior; `learned` loads the prior.
No claim is made that the cautious policy is the previous production bot.

Live learning and accepted-order identity persist in `data/runtime-v1.json`.
Practice never loads or writes that live checkpoint. Across final markets,
customer/rival behavioral learning continues while local reputation estimates
reset. `--fresh` ignores the existing checkpoint on startup and starts a new one.
Logs in `data/decisions-v1.jsonl` include each spot's best candidate, predicted
sales/profit, actual state, and chosen order. Credentials and local history stay
out of Git. Original saved data files are preserved.

Saved public live observations are replayed from `data/live-observations/` at
startup without submitting historical orders. The supplied day-nine results and
available earlier live days have been archived locally. Duplicate/out-of-order
results are ignored. A final-market checkpoint is never overwritten by old live
data. Predictions and observations are scored in order, before updating history.

```sh
python3 bot.py --predict                 # current live forecast; submits nothing
python3 tools/backtest_opponents.py      # walk-forward held-out prediction scoring
```

The read-only prediction is saved to `data/next-prediction.json`. It includes
rival attendance probabilities, location probabilities, expert-model weights,
the allowed/preferred locations, and the chosen order. The running bot archives
public results and learns during the day, before the next short order window.

The runner long-polls, identifies orders by `(mode, market, order_for_day)`, sends
before logging to disk, retries uncertain submissions using the cached order,
and immediately refreshes after phase conflicts. A policy exception invokes a
small, forecast-based fallback across all spots. Authentication failure stops the
process so an invalid key does not cause an endless request loop.

## Validation

See `model/training-report.json` and `model/validation.json` for measured results,
exact seeds, configurations, waste, poor outcomes, and decision latency. The
learned customer model reduced censor-aware held-out sales error from 4.09 to
3.03 meals per observed truck (about 26%). This measures sales prediction on
known historical lineups, not counterfactual profitability.

Before adding the user-requested small-crowd/low-price constraints, the optimizer was compared with the cautious policy on four official
30-day practice markets: two unseen seeds, each with three and seven house
rivals. It won all four, averaging $13,924.01 versus $6,832.81, with a maximum
measured decision time of 0.210 seconds. These are practice results against house
bots, not a guarantee against live adaptive competitors. Some individual days
still lose money; inventory predictions remain uncertain.

`sim.py` is a separate, approximate stress environment. It uses sequential
individual customers and pre-generated random environments for matched policies,
including customer-distribution misspecification and finite stock. Offline
results are stress signals, not official-engine results. Available rival styles
include `mixed`, `crowd`, `moving`, `undercut`, and `cycle`.

The pricing prediction models directly adapt the MIT-licensed Cheapest and
Two Bound reference strategies from
[Price Wars Merchant](https://github.com/hpi-epic/pricewars-merchant).
The two strategy classes are pinned to commit
`7981dad52034493f1f77e21e2aa297598c6ef769` in `vendor/pricewars_strategies.py`,
with the original notice in `vendor/PRICEWARS-LICENSE.md`. `pricing_models.py`
adapts their offer interface and bounds to this game. Price Wars' marketplace,
Kafka and Flink services are not needed to run this bot. The historical-data
workflow and competitive benchmarks follow the same separation of logged
observations, merchant strategies, and evaluation. Its analytics profit formula
is not used: Night Shift charges for ALL cooked meals, including spoilage.

`model/prediction-validation.json` records walk-forward scoring and final policy
tests. Prediction models are evaluated before seeing each outcome. The held-out
house-bot location log loss improved versus a repeat-location predictor; price
MAE was slightly worse than repeating the last price on these largely stable
house bots. More complex modeling is not assumed to improve every metric.
Hard simulations exercise undercutters, campers, hunters, copycats, crowd chasers,
and moving rivals; those approximations are not added to the shipped customer
prior as if they were official observations.

## Files

- `game.py`: rule parsing and customer-flow approximation.
- `learning.py`: customer/forecast learning and training-data loading.
- `strategy.py`: competitor scenarios and constrained profit optimization.
- `opponents.py`, `pricing_models.py`: contextual rival prediction and Price Wars adapter.
- `history.py`: safe, chronological local live-history replay.
- `bot.py`: API client, runtime persistence, logging, and CLI.
- `tools/train.py`, `tools/evaluate.py`: training and isolated comparisons.
- `sim.py`, `tests/`: offline stress environment and regression checks, including
  the reported day-eight loss and day-nine forecast.

[Official game documentation](https://nightshift.joinbuilders.org/kitchen-e551e5/#/welcome)
