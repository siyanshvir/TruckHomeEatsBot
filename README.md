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

Price ceilings are **$5.50 cheap, $7 standard, $8.50 premium**. Thompson's ceilings are 56% higher ($8.58, $10.92, $13.26) via `spot_price_lift`.
The optimizer can
charge less when supported by its sales forecast, with a $7.25 premium floor.
Production uses an additional downside-loss weight of 0.08 (previously 0.15);
all inventory still pays full ingredient costs, including unsold meals, and
quantities are optimized against forecast demand rather than automatically increased.
The fallback also uses the
bottom two and caps standard at $7. These are deliberate user constraints and
can reduce achievable profit compared with unrestricted optimization.

## Decisions

The bot reads `state.features` every update. With `fair_choice`, eligible trucks
split customer flow by `exp(value / 1.5)`, redistributing when stock runs out.
With `walk_over`, all local queues run first, then hungry customers may walk to
the nearest acceptable stocked location with probability `0.7 * exp(-metres/400)`
using the published campus distances. Original price limits and minimum quality
are retained. These are expected fluid flows, not the server's individual draws.
Inventory optimization caps projected uncapped demand; it approximates how our
own stock changes other trucks' leftovers.

With `salvage`, profit and downside include a refund of 25% of the ingredient
cost of unsold meals. Full cooking cost still counts against the $500 budget.
Flags enter rule fingerprints and separate customer-learning banks. Old
checkpoints retain their legacy posterior; a new combination starts broad.
While walking is enabled, crowd and rival learning continue, but local preference
fitting is withheld to avoid fitting transferred buyers as locals. Captured order
flags label realized results even if tomorrow's flags have already changed.

With walking enabled, candidate searches cover all qualities at the three
eligible locations; all six campus spots still contribute crowds and walking
flows. Under legacy rules, all six locations are searched before refinement.
Each searched spot receives equal effort. Active-night feature toggles trigger
a replan; unchanged accepted orders are not resubmitted. Missing flags use legacy
rules. Your location constraints and low price bounds are unchanged.

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
Published API crowds are used directly: they are not reduced a second time for
inactive trucks. Participation scaling at three production levels is reserved
for synthesized full-roster forecasts (explicitly marked by the offline simulator).
A bounded reputation benefit declines toward the end of a market;
a small information bonus encourages early exploration. Prices obey the low ceilings above, the game's limits, and the maximum
nonnegative customer value for that quality and reputation. Within those limits,
they do not have to undercut every rival: overflow and reputation can still matter.

`forecast.expected_customers` is the starting point. It already incorporates
weather/events. Weather scenarios are normalized around those counts, and online
calibration learns remaining forecast errors. Explicit zero forecasts stay zero.

Crowd variability is learned separately for each location and actual weather,
with sparse data shrunk toward the original +/-15% noise prior. Old checkpoints
load without retraining or discarding existing history; the new variance
statistics accumulate from newly observed forecast/result pairs. Production
evaluates 32 joint scenarios by default. Cached customer preference calculations
reuse work across crowd levels while retaining finite stock and overflow rules.
Quantity is still optimized, with no automatic stock increase after a sellout.
Production can then add a forecast-supported inventory cushion of up to 8%
(rounded up, capped at five meals). At least 60% of the extra meals must be
predicted to sell, with expected extra revenue covering at least 90% of their
ingredient cost. Expected profit and score may each fall by at most $2 or 3%
of the base expected profit, whichever is smaller; extra expected downside loss
is limited to $2. Exhausted demand and zero-food choices receive no cushion.

Plans now persist expected sales, uncapped demand, profit, and rival counts.
Completed results are matched to those plans once, including across restarts.
Changed orders are flagged rather than scored as model errors, and sellouts
remain lower bounds on demand. Inspect recent checks and API freshness from a
second terminal with `python3 status.py`. This command reads local files only;
a stale capture is not proof that the bot is still connected.

These additions were informed by the forecast calibration, reporting, and
computation-reuse ideas in [Peezy0/nightshift-food-truck](https://github.com/Peezy0/nightshift-food-truck/tree/4fe768cd99f4d85bca1ae0cbbfb420f53276397e).
They are implemented within our existing learner and customer-flow model; the
smallest-three rule, price bounds, and Price Wars rival models are retained.

## Staff, moving costs and elections

Planning subtracts what the API says we will pay: wages for the current crew
(`you.staff`), the move cost from `you.last_spot` (`rules.move_costs`, free on
the first day out), and election laws in force for the target day. Orders never
exceed `you.staff.capacity` (50 meals per employee). Price floor/cap laws change
the price customers see; junk, luxury and money laws change profit. With
the rules off, the math is unchanged. These costs are not added to the learning
regime fingerprint.

The runner casts at most one vote per election, only when a candidate's laws
add at least $30/day to our current order; otherwise it abstains, which avoids
protests against the winner's voters. Spots the model has no priors for (new
spots) are ignored rather than modeled, and the crew size and wage are left at
the API defaults. Campus-crowd laws, meat/veggie, coupons, football and riots are
not modeled.

## Crew planning and surprises

`staffing.py` chooses tomorrow's headcount and wage each night from the accepted
order: enough employees for the meals (50 each), minimizing expected wages plus
expected lost sales from a short crew. It learns the unemployed pool's asking
wages from `labor`, how often rivals paying $3+ (or 10%) more poach our
employees, and logs predicted versus actual crew size. The runner sets it with
`POST /staff` only for a clear gain, then prints the crew line each night. The
poach rate starts from a weak prior, and the retention prior for lowering wages is
an assumption.

Surprises (live from day 24): scenarios thin a random building spot's crowd 40%
on 6% of days and a random spot's by 20-40% on 10%. Hit spots are flagged and
kept out of crowd learning. Rates come from the docs, not from data.

## Learning and persistence

The accepted-order ledger `data/forecasts-v2.jsonl` now freezes code/model hashes,
rule settings, market identity, public reputations, and the exact sampled rival
lineups used by each decision. It records sales/profit quantiles, loss frequency
in those scenarios, worst-tail profit, and results conditional on named rivals
being present. These are model diagnostics, not calibrated risk guarantees.
`data/evaluation-v2.jsonl` joins later outcomes by forecast ID; practice uses
separate files prefixed `practice-`. Both ledgers are append-only and deduplicated.
Logging occurs after order acceptance, not before the time-critical POST.

Run `python3 tools/review_day.py --day 14` for a local loss breakdown. For the
experimental location-conditioned rival model, run
`python3 tools/shadow_incumbent.py --day 14 --seeds 42 43 44`. This is an offline
comparison using only prior days; it does not submit orders or change the live
model. Its own optimistic predictions are not evidence of realized profit.
See [the review process](docs/FORECASTING_PROCESS.md) for independent agent roles
and validation requirements. Production retains its existing selection and
pricing policy; no external framework or reinforcement-learning dependency is added.

Rule and market fingerprints label changed observable settings. They cannot
detect an unannounced change to hidden customer behavior by themselves. Future
data must be assessed separately when the game changes; old live results and
practice results are not proof that a candidate transfers to a new game.

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
