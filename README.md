# Truck Home Eats — Night Shift bot

Python 3.8+, standard library only. Learns customer demand and competitor habits,
then optimizes location, quality, price, and inventory for expected profit.

Configure `NIGHTSHIFT_KEY` in your existing `.env` or environment. Keep it private.

```sh
python3 bot.py                                         # live game and final
python3 bot.py --practice --seed check-a --rivals 5      # private practice
python3 -m unittest discover -s tests -v
```

Start only one live process for your truck. A GitHub push does not restart an
already-running bot. This rebuild removes the old starter/contrarian/learning
entry points; use `bot.py`.

## Decisions

Each decision evaluates all six locations and all three qualities before any
price refinement. Crowd size alone never determines location. Each spot receives
equal search effort, without an early cutoff that can skip quieter spots.

The model learns a distribution over customer budget and minimum-quality
preferences from observed sales across the market. Sequential customer-choice
modeling includes reputation, affordability, stockouts, and overflow. A sellout
is a lower bound on demand, not evidence that demand stops at the sold count.
Rival scenarios combine recent placements, historical habits, learned responses
to event crowds, prices, quality, stock, and uncertain local reputation. Only
registered rivals are modeled; no invented extra competitors are added.

For each location/quality/price, every feasible integer inventory, including
zero, competes on expected revenue minus food costs and a downside-loss penalty.
Production's contribution to campus demand is approximated at three levels and
interpolated. A bounded reputation benefit declines toward the end of a market;
a small information bonus encourages early exploration. Prices have no arbitrary
$10.50 cap and do not have to undercut rivals: overflow and local reputation can
make a higher price profitable. Prices still obey the game's limits and the
maximum nonnegative customer value for that quality and reputation.

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

The final optimizer was also compared with the cautious policy on four official
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

The undercut/cycle stress opponents are independent implementations inspired by
the documented Cheapest and Two Bound examples in
[Price Wars Merchant](https://github.com/hpi-epic/pricewars-merchant).
No third-party source or backend is bundled. Price Wars uses a different
marketplace protocol; Night Shift's official API remains the backend.

## Files

- `game.py`: rule parsing and customer-flow approximation.
- `learning.py`: customer/forecast learning and training-data loading.
- `strategy.py`: competitor scenarios and profit optimization.
- `bot.py`: API client, runtime persistence, logging, and CLI.
- `tools/train.py`, `tools/evaluate.py`: training and isolated comparisons.
- `sim.py`, `tests/`: offline stress environment and regression checks, including
  the reported day-eight loss and day-nine forecast.

[Official game documentation](https://nightshift.joinbuilders.org/kitchen-e551e5/#/welcome)
