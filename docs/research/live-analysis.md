# R1: what 31 live days say (market 1, days 2-34)

Reproduce: `python3 tools/research_live_analysis.py --data-root <data dir>` (prints every table; `--section N`). "Conv" = sold / effective arrivals (customers + walked_in - walked_out). Rival profit is estimated (revenue - ingredients + 25% salvage - wages - move); cash growth is exact.

## Findings

1. **No price collapse on a quiet spot up to about $11 premium; collapse is relative.** Alone at a spot, non-sellout, no surprise event (n=40 truck-days), conv is 0.59-0.82 for any surplus from 4 to 11 (value minus price). Premium at $9-10.5 converted 0.81 (n=12), $10.5-12 0.72 (n=15), $12+ 0.55 (n=2). Standard at $9-10.5 converted 0.72 (n=4), $10.5+ 0.57 (n=2). Real zeros happened only against cheaper rivals at reputation 50 (days 6-8: $12 sold 0-5 of 133, rivals at $5-9.5).
2. **Day 28 was a model miss, not a cliff.** $10.92 standard, rep 100 (surplus 4.7): 53 sold, conv 0.57, expected 84 (act/exp 0.63, the worst of 20), profit 223 vs 409 at $9 on day 27. The same-surplus solo days in the table convert 0.52-0.68, so the bot over-predicted standard tolerance above $10. Rain on the day adds to it.
3. **Weather and demand level, not spot, drive prediction error.** Customers / forecast expected_customers: clear 1.15 (72 spot-days), cloudy 0.99, rain 0.77; by spot only 0.98-1.02. Our act/exp: clear 1.07, cloudy 0.96, rain 0.84. The forecast's weather probabilities do not discriminate (day 14, P(clear)=0.69, total ratio 0.87; day 20, P(clear)=0.35, 1.66). Treat weather as a +/-20% demand shock, not as a forecastable signal.
4. **Sharing a spot is the biggest controllable loss.** Per truck-day, all trucks: alone 277 est profit (n=97), two trucks 62 (52), three -29 (18), four -3 (8). Same spot, alone vs two: High St 340 vs 18, Union 280 vs 105, Thompson 236 vs 71, Oval 285 vs 135. Arrivals per spot rise only ~25% (102 to 126) while trucks double.
5. **Reputation beats a price cut, and it ramps about 5 points/day while selling.** Ours: 55 (day 15) to 100 (day 27) at Thompson; rep 50 to 100 is a 1.0 to 1.3 value multiplier (premium +$4.8). Day 14: we were at North Res at rep 50 with $7.88 premium and sold 15; Cheapest Eats (home, rep 81) at $9.65 sold 59/60. Rivals that stay home (IBE at Union 24/30 days, Frontrun at High St 27/31) hold rep ~100.
6. **Best strategy so far is sit at one spot, premium about $10.3-10.7, sell out; undercutting and roaming lose.** Cash/day, days 20-34: BIg Buck (Oval, premium $10.7, 3 staff) 325; IBE (Union, premium $10.4) 304; us 277; Cheapest Eats 252; Frontrun (High St, premium $11.0, sells 52, sellout 0.26) 182; Wagyu (11 moves, cheap undercuts) about 1. We are third of six; the gap to the leaders is small and from one market only.
7. **Our worst days are the unsold ones.** Days >=15 (n=20): sellout mean profit 289 premium (n=5) / 377 standard (n=4); unsold 159 / 187 (n=5 / 6). Nine of 20 sold out, so sales for those days are lower bounds (act/exp 1.12 sellout vs 0.86 no-sellout is censoring plus rain).
8. **Walk-over is a Thompson feature.** Walked-in/day (days >=16): Thompson 15.9, High St 6.8, Oval 4.3, others under 2; Oval loses 10.3/day. Walk-ins were 21% of Thompson sales, 8.5% of all sales. Stadium and North Res lose about 5/day, mostly when empty.

## Tables

**Our days (selected; full list in section 1a)**

| day | q | price | rep | rivals | sold/meals | conv | profit |
|---|---|---|---|---|---|---|---|
| 5 | prem | 9.92 | 50 | 1 | 14/90 | 0.11 | -356 |
| 8 | prem | 12.00 | 50 | 3 | 0/60 | 0 | -330 (cost) |
| 19 | prem | 8.50 | 72 | 0 | 48/75 | 0.92 | 33 |
| 27 | std | 9.00 | 100 | 0 | 80/80 | 0.77 | 409 |
| 28 | std | 10.92 | 100 | 0 | 53/100 | 0.57 | 223 |
| 29 | std | 9.00 | 100 | 0 | 58/83 | 0.75 | 203 |
| 30 | std | 9.00 | 100 | 1 (Wagyu $5.46) | 58/79 | 0.41 | 215 |
| 32 | prem | 10.25 | 100 | 0 | 90/90 | 0.81 | 437 |

**Profit decomposition, mean of our days >=15:** revenue 614, ingredients -373, salvage +12, wages -9, taxes -7, subsidies +6, moves 0, profit 244. Best day 32 (437): 90 premium at $10.25, revenue 922, ingredients 495, flat fee +10% bonus. Worst since day 15: day 19 (33, rain, 48/75 sold) and day 14 (-344, North Res). Days 2-8 (-176 to -438) were $10-12 prices at rep 50 and 133-meal standard batches.

**Rivals (days seen, mean price, sellout rate, est profit/day, cash/day days 20-34)**

| truck | home (days) | q | price | sellout | est/day | cash/day |
|---|---|---|---|---|---|---|
| BIg Buck | Oval 19/30 | prem 24 | 10.32 | 0.40 | 222 | 325 |
| IBE | Union 24/30 | prem 30 | 9.91 | 0.50 | 246 | 304 |
| T.H.E (us) | Thompson 22/30 | 17 prem, 13 std | 9.08 | 0.43 | 114 (incl. early) | 277 |
| Cheapest Eats | North Res 20/30 | 20 prem, 10 std | 9.40 | 0.50 | 188 | 252 |
| Frontrun Fries | High St 27/31 | prem 31 | 11.01 | 0.26 | 117 | 182 |
| Wagyu Express | High St 14/24 | 19 prem, 5 std | 9.04 | 0.33 | 117 | 1 |

Rival sellouts happen at lower price than non-sellouts (Cheapest 9.16 vs 9.63; Frontrun 10.75 vs 11.10; Wagyu 8.65 vs 9.24) and with more arrivals (Frontrun 186 vs 123), usually at spots with 1.1-1.3 trucks.

**Follow/undercut episodes (only 2 in 31 days):** day 13 Wagyu onto Stadium after we left (sold out 132 standard at $9.59 on a game day); day 30 Wagyu onto Thompson at $5.46 standard, sold out 67/67 (est about +$100). Our result that day: 58/79, profit 215 vs 203 alone on day 29. Clear weather (customers 1.21x forecast) offset the loss, so the undercut cost us about nothing and earned Wagyu little.

**Competition (all trucks)**

| trucks at spot | truck-days | sold/truck | est profit | sellout rate |
|---|---|---|---|---|
| 1 | 97 | 67.9 | 277 | 0.52 |
| 2 | 52 | 49.9 | 62 | 0.29 |
| 3 | 18 | 39.6 | -29 | 0.11 |

**Prediction (20 days):** profit act-exp by crowd shortfall: ok +46 (n=8), over +32 (4), short (<0.9 of forecast) -101 (8). By price: <=8.5 -11 (11), 8.5-9.5 -1 (5), >9.5 -46 (4).

## Implications for the final bot

1. Stay at one spot all market; do not move to chase forecast crowds. Reputation resets to 50 each market and rises about 5 points/day while selling, so day 1-10 prices must be set at rep 50 value, then raised as rep climbs.
2. Choose the spot with the fewest likely rivals; each extra truck cost about $200 per truck-day. Rivals are habitual (IBE, Frontrun, Cheapest, BIg Buck each at home 63-87% of days), so predict occupancy from history and avoid homes.
3. Do not set standard above about $10 or premium above about $11 without a conversion check; day 28 earned about $190 less than the $9 days. Cap price-response tolerance by quality in the model.
4. Never rely on a cheaper rival at your spot being a problem if reputation is higher: compare surplus (value minus price), not price. A reputation lead of 30 points is worth about $3 on premium.
5. Size meals as a mixture over weather (+/-20%); the budget cap (premium 90, standard 133) already binds on good days, so the upside on clear days is limited by cash, not demand.
6. Do not copy Wagyu: low-price roaming earned about zero cash. Premium about $10.3-10.7 at a protected home spot (IBE, BIg Buck) earned the most.
7. Treat sellouts as demand lower bounds in the learner; they were 45% of our days.
8. Thompson's walk-in stream (16/day) is a real, cheap bonus; keep a spot adjacent to a busy neighbour (Oval) if choosing a quiet one.

## Limits

One market, 31 observed days (days 1, 3, 4 and a sparse day 8 missing), 6 trucks; none of our prices were randomised, so price effects are confounded with weather and day. Walk fields exist from day 16, crew and wages from day 22; rival profit is an estimate that ignores taxes and fines (flat fee and curfew from day 27). Reputation is read from the previous day's file (50 where missing). `hungry` is inferred as unserved after walking. Rival spot choice is endogenous, so the competition effect mixes selection and substitution. Surprises (free food, fire drill, protest) are excluded from conversion tables only. Standard-vs-premium rival comparisons have n of 6-9.
