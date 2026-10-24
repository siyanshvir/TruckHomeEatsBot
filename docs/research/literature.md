# R2: literature and outside work

Evidence grades: **S** = strong (theorem or replicated), **M** = moderate (simulation/one study), **F** = folklore/analogy. Sources were read via search abstracts and one repo page, not full texts; constants are unchecked. Test sketches assume an offline simulator with matched seeds.

## 1. Censored demand
**1. Huh, Levi, Rusmevichientong, Orlin (2011), Kaplan-Meier newsvendor.** https://pubsonline.informs.org/doi/fpi/10.1287/opre.1100.0906
Nonparametric adaptive ordering from sales-only data; Kaplan-Meier treats sellouts as right-censored observations. Converges for discrete demand and is far more robust than naive "sales = demand" or parametric fits. [S]
*Implies:* treat each sellout day as censored (demand >= sold), not an observation. Estimate the demand distribution per (spot, quality, price-bucket) with KM, then size meals at the critical fractile (price - cost)/(price - cost + 0.25*cost refund-adjusted underage). Never shrink toward realised sales.
*Test:* simulate demand with known law, 40% censoring, compare mean-sales, Tobit and KM estimates of quantile q*; measure profit regret of the resulting order.

**2. Jain, Rudi, Wang, "Stock-Out Timing Is (Almost) All You Need" (Oper. Res.).** https://pubsonline.informs.org/doi/references/10.1287/opre.2014.1326
Argues the time of stockout carries most of the information in censored data. [M; abstract-level]
*Implies:* log the time/queue position when stock ran out, if the API exposes it. An early sellout at a known rate gives a much tighter demand lower bound than "sold N".
*Test:* in the offline sim, give the estimator stockout time vs only count; compare quantile error.

## 2. Pricing with learning
**3. Besbes and Zeevi (2009), pricing without knowing demand.** https://repository.upenn.edu/handle/20.500.14332/42019
Regret bounds: a short, deliberate price-experimentation phase then exploit is near-optimal; learning has a quantifiable cost. [S, but horizon-large asymptotics]
*Implies:* 30 days x 6 spots x 3 qualities is a tiny horizon; exploration must be targeted (few price points, around the current model optimum), not broad. Cap exploration spend as a fixed fraction of expected profit.
*Test:* matched-seed runs with exploration budget 0/1/3 off-optimum days per market.

**4. Broder and Rusmevichientong (2012), general parametric choice model.** https://pubsonline.informs.org/doi/fpi/10.1287/opre.1120.1057 (via https://ideas.repec.org/a/inm/oropre/v60y2012i4p965-980.html)
Regret is order sqrt(T) in general, order log(T) when demand curves are well separated; "uninformative prices" (where curves cross) are the obstacle. [S]
*Implies:* our logit/softmax value model is parametric, so MLE plus certainty-equivalent pricing is near-optimal and extra dithering is waste. Price at the model optimum, but avoid prices where alternative fitted models agree (they teach nothing).
*Test:* compare MLE-greedy to Thompson-perturbed price on matched seeds; expect greedy to win at 30 days.

**5. Thompson sampling for dynamic pricing (arXiv 1802.03050) and Ferreira, Simchi-Levi, Wang (2018).** https://arxiv.org/pdf/1802.03050 ; https://hdl.handle.net/1721.1/125757
TS with inventory constraints balances price experimentation and revenue; works well empirically. [M]
*Implies:* if we want exploration, sample demand parameters from the posterior and plan once per scenario (we already do Monte-Carlo): one posterior draw per scenario batch is "free" TS.
*Test:* swap posterior mean for posterior draw in the planner; compare profit and variance.

## 3. Bandits for bounded tuning
**6. Garivier and Moulines, discounted/sliding-window UCB.** https://www.arxiv.org/abs/0805.3415
For abruptly changing rewards, discounting or windowing is needed; regret scales with sqrt(T * breakpoints). [S]
*Implies:* our tuning bandit must discount (gamma ~ 0.9-0.95) because rivals adapt, but market reset (20 independent markets) argues for resetting the bandit per market and using cross-market priors only as the prior mean.
*Test:* simulate arms whose payoff shifts when a rival adapts; compare stationary vs discounted vs reset on 30-day horizons.

**7. Discounted/sliding-window Thompson sampling (arXiv 2305.10718).** https://arxiv.org/pdf/2305.10718
Gives regret guarantees for discounted TS with uninformative priors. [M; new]
*Implies:* with ~30 pulls per market, priors dominate: use informative priors from the learned model, bound arms to a narrow trust region, and do not let bandit choose outside the planner's top-k.
*Test:* bandit over 3-5 knobs with noisy payoff std ~ daily profit std; check regret vs fixed default (the realistic baseline). Expect small gain; treat as folklore until shown.

## 4. Competition
**8. Calvano, Calzolari, Denicolo, Pastorello (AER 2020).** https://art.torvergata.it/bitstream/2108/303883/1/aer.20190623.pdf
Q-learning pricers in logit competition learn supracompetitive prices sustained by punish/forgive patterns; critics say this is not canonical collusion. [M; simulation, long horizons of millions of periods]
*Implies:* with 30 days and heterogeneous rivals, do not count on tacit collusion; but a stable "follow price, don't undercut" rival is exploitable: hold higher price when rivals' recent prices are sticky. Detect undercut-and-revert patterns before leading a price cut.
*Test:* build scripted rivals (always-follow, tit-for-tat, fixed), check that our learned rival model identifies them within 5 days and that profit beats always-best-response.

**9. Price Wars platform (HPI).** https://github.com/hpi-epic/pricewars
Open marketplace testbed for repricing merchants; archived 2020; no results reported on the page I read. [F for our purposes]
*Implies:* mainly a pattern: benchmark strategies by running many merchants head-to-head over fixed durations with fixed seeds. We already do this offline; add a rival zoo (cheapest, follower, cycle).
*Test:* run our planner against the zoo and report per-rival profit delta.

## 5. Location
**10. Hotelling-type agglomeration (1995 working paper) and related empirical restaurant studies.** https://ideas.repec.org/p/ivi/wpasec/1995-17.html
Clustering can be an equilibrium when consumers weigh several stores; challengers in chain studies tended to sit near leaders for agglomeration benefits while leaders differentiate. [M-F, analogy only]
*Implies:* walk-over (0.7*exp(-d/400)) is a spillover, so a quiet spot near a busy one inherits overflow when the busy one sells out. Under the 3-smallest-spot rule, prefer the small spot adjacent to a big rival cluster that runs short, with meals sized to the overflow.
*Test:* fit overflow share to observed sellout events of neighbours; counterfactual sim with/without walk-over in the value.

## 6. Reputation and quality
**11. "The Effect of Price on Firm Reputation" (Management Science 2021, NBER w27405).** https://www.nber.org/papers/w27405.pdf
Price acts as a reference point for quality expectation; price increases hurt ratings, especially for first-time customers. [M; field data, not a game like ours]
*Implies:* if the game's reputation moves with a price-quality gap, an overpriced premium meal may cost reputation as well as demand; value-for-money should be scored net of price. Check whether our reputation model has a price term.
*Test:* regress reputation change on quality and price from logs; if no price term, keep it out.

## 7. Tournament risk
**12. Hvide (2002) and review of risk-taking in tournaments.** https://emerald.com/insight/content/doi/10.1108/JM2-09-2018-0145/full/html
Tournament payoffs (rank-based) induce risk seeking for trailing players. [M]
*Implies:* the final ranks on average profit over 20 markets, a sum of many independent draws: variance mostly averages out, so near risk neutrality is right for the *bot* objective. Rank-incentive risk seeking applies only to leaders vs trailers in a rank contest with a threshold; with one winner, a trailing team gains from variance, but 20 markets limit how much variance is available.
*Test:* simulate 20-market totals for risk weight 0/0.08/0.15; compare mean and P(top rank) against a rival with equal mean.

**13. Schweitzer and Cachon (2000) pull-to-center; replication (PMC8863238).** https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8863238/
Human newsvendors order too few for high-margin items, too many for low-margin; recency bias contributes. [S for humans]
*Implies:* a bot that adapts quantity to the last observation inherits this bias. Use the full-history critical fractile, not the last day. Rival bots built by simple heuristics may show it (undersupply on premium).
*Test:* check our past meal counts vs fractile on logs for drift toward yesterday's sales.

## 8. Calibration
**14. Gibbs and Candes, Adaptive Conformal Inference (NeurIPS 2021).** https://arxiv.org/pdf/2106.00170v2
Online wrapper keeps long-run interval coverage under distribution shift by adjusting the miscoverage level with one step-size parameter; JMLR follow-up tunes the step size. [S for coverage, not profit]
*Implies:* use it for intervals (e.g., demand quantile for meal count), not to reshape the mean. With ~30 days per market, intervals will be wide; blend with a prior (shrinkage is standard statistics; I did not source it) and keep calibrator effects bounded.
*Test:* replay logged forecast errors and see whether ACI quantiles hit 80% coverage in 10-day blocks vs fixed residual quantiles.

## 9. Other team's bot
**15. Peezy0/nightshift-food-truck.** https://github.com/Peezy0/nightshift-food-truck
Profit-agent simulates the queue at every allowed spot, estimates rival parking and price, scores by expected profit, loss probability and reputation; tests meal counts and cooks leaner after losing days; caps production at 50; has a heuristic fallback and manual steering file (avoid/prefer spots, bonus, caution). Plans in ~50 ms. [F: read from README summary only]
*Differences:* ours has learned mixtures for rivals, a calibrator, a bandit and Monte-Carlo scenarios; theirs is simpler with human knobs and loss-streak throttling. Their "cook leaner after losses" is a reactive rule the newsvendor logic does not support when losses are noise.
*Test:* implement their heuristic as a rival in the zoo.

## Top 8 implications ranked by expected impact on final average profit
1. Treat sellouts as censored (KM/stock-out timing) and size meals at the critical fractile. Direct profit lever; strong theory.
2. Keep near risk neutrality for the 20-market average; reduce downside weight only if sims show a mean gain.
3. Price at the model optimum; limit exploration to a small, fixed budget (short horizon).
4. Identify follower/undercutter rivals early and exploit stickiness; do not lead price cuts into followers.
5. Use overflow from busy neighbours when choosing among the permitted small spots.
6. Reset the bandit per market, narrow arms, informative priors; keep a kill switch.
7. Use conformal intervals for quantity quantiles, not for the mean.
8. Check for a price term in reputation; avoid pricing premium above the value-for-money frontier.

## What the literature says we are probably doing wrong
- Any use of last-day sales or sell-through for tomorrow's quantity (pull-to-center, censoring bias).
- Learning too broadly: regret theory says the 30-day horizon cannot pay for wide exploration or a bandit with many arms.
- Over-weighting downside (0.08-0.15) if the metric is the mean: it trades mean for variance that averages out over 20 markets.
- Expecting tacit collusion: evidence comes from million-round games; here only sticky rivals are exploitable.
- Caveat: nearly all sources are asymptotic or single-product; none models fair-choice softmax with walk-over, so ordinal advice only.
