# Research, synthesis and thesis for the final — design

## Why
The club lead asked for research, synthesis and a thesis for the final (and the current model). The
final ranks **average profit over 20 seeded markets × 30 days**, reputation resets to 50 each market,
every truck plays every market, 2-second window. We have ~5 hours. The work must end in a decision
that changes (or deliberately keeps) the bot, with evidence.

## Deliverables
1. **Report** (`docs/research/REPORT.md`, also published as a private page): findings, synthesis, thesis.
2. **Experiment plan and results** (`docs/research/experiments.md` + JSON in `data/`): ranked
   experiments, each run offline (matched seeds) or on private practice, with results.
3. **Decision memo** (end of report): which settings/models the final bot uses and why, what we reject.
4. **Incorporation**: only after the user approves the decision memo. Changes go to `finals` first
   (and to `main` only if the user says so).

## Streams (parallel, read-only except their own docs/tool files)
- **R1 live-data analysis**: what 35 live days say. Price response by price/quality/spot, prediction
  error structure (weather, price level, sellouts), rival timelines (who goes where, at what price,
  follow/undercut episodes), profit decomposition, what drove the best and worst days.
- **R2 literature and outside work**: methods that matter here, with implications for our design:
  censored-demand estimation (newsvendor with sellouts), dynamic pricing with learning, bandits for
  bounded tuning, repeated-pricing competition and tacit reactions, reputation dynamics, objective for
  an average-over-markets tournament (risk neutrality), plus other teams' public bots.
- **R3 experiments**: offline matched-seed sweeps of the knobs that matter (risk, price headroom,
  location breadth, calibrator on/off, crew spare, meal-sizing rule, follower rivals, reputation reset)
  and a small number of official practice markets. Results with per-market regressions.
- **Synthesis (lead)**: one model of the game and 5–8 falsifiable thesis statements.

## Thesis format
Each statement: claim; mechanism; evidence (data/literature/experiment); how it could be wrong; the
bot behavior it implies; verdict (supported / weak / rejected).

## Decision rule
A change is adopted for the final only if it (a) wins on matched offline seeds on average, (b) does not
worsen tail losses materially, (c) is supported by at least one independent source (data, literature or
practice), (d) keeps latency < 1 s, and (e) has a kill switch. Otherwise it is rejected or deferred.

## Gate
The user reviews the decision memo before any bot change. Research itself never contacts live orders.
