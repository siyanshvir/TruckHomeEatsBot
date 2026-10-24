# Forecasting and review process

This process improves forecasts and explains decisions while preserving the current policy: consider the three smallest **tomorrow customer forecasts**, prefer the smallest two unless the existing profit hurdle is cleared, keep the existing low price limits, and obey the $500 ingredient budget. A review does not by itself change the live policy or submit an order.

## Day 14: what the records establish

The bot ordered 84 premium meals at North Res for $7.88. Its saved decision predicted 76.74 sales and $142.73 profit. It sold 15, with $118.20 revenue, $462 ingredient cost, 69 spoiled meals, and a $343.80 loss. Break-even required 59 sales.

The inventory cushion added two meals above the optimizer's 82-meal choice. Those two meals cost $11; removing the cushion alone would still have left a $332.80 loss. North Res had 83 customers against a forecast of 101. The approximately 18% crowd shortfall cannot by itself explain the approximately 80% sales shortfall.

Cheapest Eats returned to North Res with premium food at $9.65 and 60 meals, selling 59. Its pre-day reputation there was 80.84 versus our 50. Under the current customer utility formula, that reputation advantage was worth about $2.96 for premium food, exceeding our $1.77 discount. Being cheaper did not make our offer more attractive. Matching that particular rival's value would have required a price below about $6.69, below our current premium floor. This is evidence to review conditional competition forecasts and downside risk, rather than automatically lowering prices or globally cutting quantities.

## Development roles

When requested, subagents divide development and review work:

- **Predictor:** checks tomorrow's weather, events, customer forecasts, opponent location probabilities, and opponent offers conditional on location. Uses all available prior days, with suitable context and forgetting, rather than copying yesterday's offer.
- **Risk validator:** independently checks public reputation, ingredient costs, inventory, conditional demand, loss probability, and losses in adverse scenarios. Challenges proposals against preserved policy constraints and compares alternatives on unseen data.
- **Coordinator:** reconciles evidence, runs integration checks, records the comparison, and reports which changes are supported. Keeps an unproven candidate separate from the live model until the promotion checks pass.

These are development and review roles. The bot runs its Python forecasting and optimization code at runtime; it does not run live LLM subagents every night. This document does not create a background monitor or a scheduled autonomous review.

## Data and prediction ledger

Existing public API snapshots, saved decisions, and checkpoints remain the source of observations and replay state. Record each prediction **before** its result is available. Keep raw observations separate from derived features, predictions, and evaluations. Do not log API keys. Use mode, market, day, and model version to distinguish records, and deduplicate repeated API updates.

The target logging format is append-only JSON Lines:

| File | Record contents |
| --- | --- |
| `data/forecasts-v2.jsonl` | Record/schema version; timestamp; mode/market/target day; last observed day; source API version; model/code version; forecast weather/events/customer counts; exact public pre-day reputations; eligible locations; opponent attendance and location probabilities; conditional price/quality/stock summaries; candidate orders; demand quantiles; expected sales/profit/waste; probability of loss; downside-tail summary; chosen order; submission state. |
| `data/evaluation-v2.jsonl` | Matching prediction identifier; observed day/weather/customer counts; actual rival locations/offers and own order; order-match flag; actual sales/profit/waste; forecast errors; interval coverage; loss and sellout flags; censoring status; evaluation timestamp. |

Treat these as a schema contract for the logging work, not a claim that every field is already available. Unknown fields should be explicit nulls rather than invented estimates. Log API acceptance separately from planning. A sellout reveals only a lower bound on uncapped demand; do not train or score it as proof that demand ended at the cooked quantity.

The current implementation writes frozen model/scenario records and joined
accepted-order evaluations in these files, with separate `practice-` prefixes
for private practice. Per-rival probabilities and candidate rankings are in
`decision_diagnostics`; raw sampled offers and customer models are retained.
Scenario quantiles and conditional risk are descriptive, not calibrated
intervals. Interval-coverage scoring and automatic candidate promotion are not
implemented. The offline candidate remains outside production.

## Unknown midnight changes

The organizers have warned that the game will change, without specifying which
mechanics. Do not infer that only the season resets or that old fitted patterns
will transfer. Forecasts include a rule-settings fingerprint and market identity,
so observably different games can be separated in analysis. Hidden behavior can
change without those fingerprints changing; detect that through future forecast
errors and fresh practice/live evidence. These labels do not automatically reset
the production learner. Avoid team-name-specific rules and tuning to Day 14.

Competition evaluation must include **who arrives and what they offer**, not only the rival count. One high-reputation premium camper can matter more than several unattractive offers. Preserve a rival's observed home-location quality and price history when producing conditional scenarios; a special-event move must not erase that evidence. Use exact published reputation when available and label uncertainty for unknown future offers.

## Training and validation

1. Reconstruct each day's information cutoff from snapshots. Only observations available before that order may enter its features or fitted model.
2. Predict the next day, save the full prediction, then reveal the result and update the learner. Apply this chronological, prediction-before-observation procedure to all replay evaluation. Do not use random row splits that mix later days into training.
3. Keep comparison baselines: existing production model, a simple previous-day offer forecast, and a contextual conditional-offer candidate. Compare crowd, opponent, and own-sales errors separately so one source of error is not mistaken for another.
4. Evaluate interval coverage, probability calibration, profit, waste, loss frequency, and adverse-tail losses. Handle sellout censoring explicitly. Report sample sizes and uncertainty; roughly 14 live days, with fewer usable saved forecasts for some locations, cannot establish robust superiority.
5. Use Day 14 as a regression case and Day 13 as a high-demand counterexample. Never tune solely to either example. Check ordinary days, events, home returns, absent rivals, and high-reputation competition across the chronological record.
6. Run fresh official practice comparisons and matched brutal and moving simulations with held-out seeds. Keep simulated observations out of the live checkpoint. The approximate simulator is a stress test, not proof of performance in the official engine.

## Promotion criteria

A candidate is ready for promotion only after its behavior is reviewable and all of the following have been checked:

- Existing smallest-three selection rules, low price limits, and cost/budget constraints still hold.
- Prediction code and ledger tests pass, including information cutoffs, duplicate updates, sellout censoring, and accepted-order matching.
- Fresh official practice and held-out brutal/moving comparisons show acceptable profit, waste, and downside behavior. Report individual market regressions as well as averages; use several seeds rather than promoting from one winning run.
- The Day 14 case is better represented by the forecast distribution, without requiring a blanket low inventory cap or fitting to its realized rival order.
- The decision latency stays suitable for the order window, and the coordinator reports limitations and rollback/version information.

Choose evaluation thresholds before reviewing the final held-out comparison. A tradeoff that raises average profit but materially worsens tail losses should be reported, not hidden by the average. No result guarantees live profit, especially with sparse data and opponents changing their code.

## References

- User-supplied research query: [Python future prediction models in competition settings](https://www.google.com/search?q=pythn+future+prediction+model+in+competition+setting). This is a discovery starting point, not evidence that a specific algorithm fits this game.
- User-supplied forecasting overview: [Forecasting: A Practical Guide](https://medium.com/@alan.vourch/forecasting-a-practical-guide-6173f421c1ed). General background; validate any proposed technique against the actual game's information and cost rules.
- Official scikit-learn example: [Lagged features for time series forecasting](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html). Illustrates evaluation with chronological splits and prediction intervals; adding this library is not required by this process.
- User-supplied introductory pipeline: [Build a predictive model using Python](https://www.analyticsvidhya.com/blog/2015/09/build-predictive-model-10-minutes-python/). Use the preparation/baseline/evaluation structure; replace random row splits with chronological evaluation for this game.
- [QuantEcon game-theory notebooks](https://github.com/QuantEcon/game-theory-notebooks) provide best-response and fictitious-play examples. Our existing sampled best-response optimizer already uses related concepts; no framework is imported.
- [GitHub pricing-optimization topic](https://github.com/topics/pricing-optimization) is a discovery index, not a validated Night Shift strategy. Price Wars hypotheses are already part of the rival model.
