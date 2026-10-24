# Research, synthesis and thesis: what wins the final

Sources: live-data analysis (`live-analysis.md`, 31 live days), literature (`literature.md`, 15 sources),
offline sweeps (`tools/research_sweep.py`, 14–15 paired 20-day markets per configuration, approximate
simulator, reputation resets each market), two private official practice markets, and the live results.

## 1. Synthesis: how the game works
1. **Profit is set by who else is at your spot.** Per truck-day: alone about $277, with one other truck $62,
   with two $-29 (n=97/52/18, confounded by selection). Rivals are habitual: each sits at its home spot
   63–87% of days, so occupancy is predictable from history.
2. **Price tolerance is relative to value, and value depends on reputation.** Customer value is
   quality value × (0.7 + 0.006 × reputation) minus price. Alone at a spot, premium near $10–11 converted
   well at reputation 100; at reputation 50 the same price is far above value (live days 6–8 sold 0–5).
   The final resets reputation every market, so early-market prices must follow value, not the live tuning.
3. **Reputation compounds.** It rose about 5 points a day while selling premium (55 to 100 over days 15–27),
   and a 30-point lead is worth about $3 on a premium meal. Staying at one spot earns that return.
4. **Demand is noisy by weather, not by spot.** Customers over forecast: clear 1.15, cloudy 0.99, rain 0.77.
   The forecast's weather odds did not separate good days from bad. Size orders over a ±20% shock.
5. **Sellouts hide demand.** 45% of our days sold out, so sales are lower bounds; unsold days averaged
   $159–187 against $289–377 for sellout days.
6. **Rival reactions are rare and mild.** Two follow/undercut episodes in 31 days (Wagyu on day 13 and 30);
   the day-30 undercut at $5.46 cost us about nothing. Low-price roaming earned Wagyu about $0 in cash.
7. **Walk-over favors a small spot next to a busy one** (Thompson got about 16 walk-ins a day, 21% of sales).

## 2. Thesis statements (claim, evidence, how it could be wrong, verdict)
| # | Claim | Evidence | Could be wrong if | Verdict |
|---|---|---|---|---|
| T1 | Expected profit is driven mainly by rival occupancy; choose the spot with the fewest likely rivals and stay when reputation is high | live: $277/$62/-$29 by trucks at spot; rivals' homes 63–87% | selection effects (good spots attract trucks) | Supported |
| T2 | A learned price-response correction beats a fixed model | sweep: calibration off −$437 (se 229, about 1.9 se), waste 620 vs 386 meals; live day 28 predicted 84, sold 53 | calibration chases noise on few days | Supported |
| T3 | The adaptive profile beats the baseline | sweep: +$3,160 (se 588) over 14 paired markets, 0 of 14 wins for the baseline; practice 30 days, 7 rivals: $10,108 vs $8,465 | simulator not the official engine; house-bot rivals | Supported, with the simulator caveat |
| T4 | Risk weight and price headroom barely matter for mean profit | sweep: risk .03/.30 and price ±10% all within ~1 se of default (−$11 to −$33) | effect appears only in other rival mixes | Supported (no action) |
| T5 | Wide location breadth hurts | breadth 1.0 −$253 (se 117, about 2.2 se); breadth 0 −$18 | one test, small sample | Weak |
| T6 | Risk-seeking when behind lowers mean profit | posture on vs off, 18 paired markets: −$140 (se about 69), 6 wins 6 losses; worst in brutal 7-rival markets (−$600, −$806) | small sample | Weak but sufficient to keep it off in the final |
| T7 | Orders should be sized to learned demand, not to a crew or last-day sales | crew cap locked us at 50 meals; literature: censored-demand newsvendor | demand estimates are noisy | Supported (cap removed) |
| T8 | A reactive "cook leaner after losing days" rule is noise-chasing | literature (recency bias); rival bot uses it | losing streaks signal a real change | Rejected |

## 3. Decision memo (for approval; nothing is changed yet)
Keep for the final:
1. The adaptive final profile as built (calibrator on, competition model on, all spots searched, bounded tuner).
2. The calibrator: removing it lost about $437 and wasted far more food (moderate evidence, about 1.9 se).
3. Posture neutral in the final (already the default); mild in the live game only.
4. The crew model with no hard crew cap and data-driven wage and headcount ranges.
Proposed small changes (each cheap, each reversible, each needing approval):
- **A. Drop the `breadth = 1.0` arm** (keep 0 and 0.5). Weak evidence, low risk.
- **B. Initialize the final's first market with a reputation-aware price check**: at reputation near 50 the planner must price from value (already done by the calibrator ratio); add a practice test of day 1–10 pricing before the final.
- **C. Use the full-history critical fractile for meal counts** (Kaplan-Meier style) as an offline experiment first; adopt only if it wins on matched seeds.
Rejected: Wagyu-style low-price roaming, last-day throttling, expecting tacit collusion, risk-seeking in the final.

Also from R3: the all-spots search without the calibrator and tuner (`unrestricted`) earned $7,385 and its worst market was $842 against $4,872; final decisions average about 1.0 s against 0.67 s for the baseline offline. The tuner's learning gain was not measured (fixed knobs); a controller-on arm and about 10 seeds for breadth and calibrator are the next experiment.

## 4. Limits
One live market (31 days), small per-pair samples, simulator not the official engine, rival profit estimated.
Several sweep comparisons are within noise; "no effect" means no effect we can detect with 14 markets.
