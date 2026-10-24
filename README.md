# TruckHomeEatsBot
Bot for Builders: a Night Shift food-truck bot. Python 3.8+, standard library only.

```
echo "NIGHTSHIFT_KEY=fk_your_key_here" > .env   # never commit this file
python3 bot.py                                  # live game / final
python3 bot.py --practice --seed s1 --rivals 3  # one practice market
python3 bot.py --practice --starter             # the original starter strategy, for comparison
python3 sim.py                                  # offline simulator: starter vs strategy
```

## Files
- `bot.py`: polling loop and order sending; wraps the strategy with a safe fallback order.
- `strategy.py`: the decision engine (`Brain`).
- `sim.py`: offline simulator of the rules, calibrated against practice games, used for tuning.

## Strategy
Each night the brain:
1. reads yesterday's results: where every rival parked, their price, quality, meals and sales;
2. fits the hidden customer price limits and quality minimums per spot to what actually sold,
   and corrects the crowd forecast for weather;
3. predicts tomorrow's rivals from their habits (steady rivals exactly, adaptive ones with a price spread);
4. plays out the customer-choice rules (best value wins, sold-out trucks spill to second choices)
   for every spot, quality and price;
5. cooks the number of meals that maximizes expected profit (spoilage vs lost sales) and adds the
   long-term value of the reputation change.

## Results (practice seeds s1-s5, 3 house rivals, 30 days)
| Version | Average profit |
|---|---|
| Starter bot | -$2,999 |
| Strategy | $11,822 |
