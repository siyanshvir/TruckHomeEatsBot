# Final playbook: start to end

## Before 8:30 AM
1. Plug in the laptop and run `caffeinate -dims` in a spare terminal (the bot stops if the machine sleeps or the battery dies).
2. One bot only. After the last code change, restart it during a day phase, right after an order is accepted:
   `cd ~/Downloads/TruckHomeEatsBot && python3 bot.py --profile final`
3. Expect `Profile: FINAL (adaptive)`. Tests: `python3 -m unittest discover -s tests -q` (248 pass).

## What the bot does in each market (20 markets x 30 days, every market starts at $0, reputation 50)
- **Night 0 (before day 1):** reads rules and features, searches every open spot (no location or price caps), prices from customer value at reputation 50, orders, then sets the crew to match the order (one employee per 50 meals).
- **Days 1-10, build a home:** stays at the spot with the best expected profit and fewest rivals; reputation rises about 5 points a day while selling, and prices follow value. Roaming, upside-seeking and exploration stay off until reputation is established (85+).
- **Days 10-25, exploit:** sizes meals to learned demand (sellouts are lower bounds), takes big-crowd days (game days: the Stadium when it clearly beats home), keeps a calibrated price response, avoids spots where rivals are likely (learned habits, followers, phantom rival).
- **Last days:** nothing special except the complaint model, which only considers filing when the ban cost is small (rare; at most one per 3 days; off with `NIGHTSHIFT_COMPLAINTS=off`).
- **Every night, after the order is accepted:** crew plan (data-driven wage and headcount), vote analysis (abstains unless a candidate clearly pays), riot/coupon/law adjustments, learning from the completed day (surprise and protest days are kept out of crowd learning).
- **New market:** reputation, cash and votes reset; the bot keeps customer, rival, calibration and crew learning. First final state resets the tuner, competition model and rule residuals once.
- **Showcase (top 4 only):** if `you` is null the bot waits and sends nothing.

## Rules that are not live yet (handled if they switch on)
New spots (registered and orderable), riots (cost model, 403/wrecked skip), coupons (live from day 61): drops on us are planned for from `you.coupons` and the tuner ignores `coupon_loss`; we drop one ourselves only rarely (`coupon_play.py`: final only, at most once per market, day 10 or later, on a rival whose cash is within 85-125% of ours, and only when its expected loss is at least 3x our cost; `NIGHTSHIFT_COUPONS=off` disables; each drop costs double the last: $20, $40, $80), meat or veggie (the order gets a `food` field chosen from the forecast diets; demand is not diet-modeled yet), planned counts (not used yet). Unknown rules: ignored safely; unexplained profit terms are learned by spot.

## Kill switches
`NIGHTSHIFT_PROFILE=baseline` (whole adaptive profile off), `NIGHTSHIFT_POSTURE=off`, `NIGHTSHIFT_COMPLAINTS=off`, or `data/profile.json` with `{"profile":"baseline"}`, `{"posture":"off"}`, `{"complaints":"off"}`. Details in `docs/FINALS.md`.

## If something looks wrong
- `Policy error (...)` lines name the file and line; the bot still orders (fallback). Send me the line.
- No order line for a night: check the process is running and the API state is fresh (`python3 status.py`).
- Rules changed: stop the bot, tell me, restart after my update.
- Rollback: tags `live-baseline-day29` and `main-before-final-merge`.
