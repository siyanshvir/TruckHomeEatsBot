"""Self-adjusting posture: how much risk and exploration the planner takes, from how the field is doing.

Everything is read from the API each night (every truck's cash). Nothing is set by hand:
- pressure: the fastest rival's recent daily earnings exceed ours. The further behind, the lower the
  weight on expected losses, the more weight on upside variance, and the more a quiet spot we do not
  usually use is worth trying.
- lead: we out-earn every rival. Then risk weight rises and exploration falls, so we protect a home
  spot where our reputation already pays.
All effects are bounded, scale in smoothly as evidence accumulates, and switch off with
NIGHTSHIFT_POSTURE=off or data/profile.json {"posture": "off"}.
"""
WINDOW = 7            # days of cash history used for recent daily earnings
MIN_DAYS = 4          # below this much history the posture stays neutral
FLOOR = 60.           # dollars/day: scale below which gaps are noise
MAX_UPSIDE = .5       # weight on E[max(0, profit - mean)] at full pressure
MAX_EXPLORE = 40.     # dollars of bonus for a quiet, rarely used spot at full pressure
NEUTRAL = {'pressure': 0., 'lead': 0., 'risk_scale': 1., 'upside': 0., 'explore': 0.}


def _rates(history, day):
    """Recent daily earnings per truck from {day: {truck: cash}}."""
    days = sorted(d for d in history if isinstance(d, int) and d <= day)
    if len(days) < 2:
        return {}, 0
    last = days[-1]
    base = next((d for d in days if d >= last - WINDOW), days[0])
    span = max(1, last - base)
    start, end = history[base], history[last]
    return {t: (end[t] - start[t]) / span for t in end if t in start}, span


def enabled(root=None):
    """Kill switch: NIGHTSHIFT_POSTURE=off or data/profile.json {"posture": "off"}."""
    import json, os
    from pathlib import Path
    if os.environ.get('NIGHTSHIFT_POSTURE') == 'off':
        return False
    try:
        path = Path(root) if root else Path(__file__).resolve().parent / 'data' / 'profile.json'
        return json.loads(path.read_text()).get('posture') != 'off'
    except (OSError, ValueError, AttributeError):
        return True


def read(state, history, me):
    """Posture for tonight. `history` maps day -> {truck id: cash}, newest included."""
    day = int(state.get('day') or 0)
    rates, span = _rates(history, day)
    if span < MIN_DAYS - 1 or me not in rates or len(rates) < 2:
        return dict(NEUTRAL)
    ours = rates[me]
    best = max(r for t, r in rates.items() if t != me)
    trust = min(1., span / float(WINDOW))                    # fewer days, less weight
    scale = max(abs(best), abs(ours), FLOOR)
    pressure = trust * max(0., min(1., (best - ours) / scale))
    if state.get('mode') not in ('final', 'showcase'):
        # One shared market to the end (the live game): to overtake the cash leader we must out-earn
        # their pace by gap / days_left. In the final each market is scored on its own average profit,
        # so only the pace comparison above applies there.
        cash = history.get(max(d for d in history if d <= day), {})
        leader = max((t for t in cash if t != me), key=lambda t: cash[t], default=None)
        left = int(state.get('days') or 0) - day
        if leader and left > 0 and cash.get(me) is not None and cash[leader] > cash[me]:
            needed = rates.get(leader, 0.) + (cash[leader] - cash[me]) / left
            pressure = max(pressure, trust * max(0., min(1., (needed - ours) / max(needed, FLOOR))))
    lead = trust * max(0., min(1., (ours - best) / scale))
    return {'pressure': pressure, 'lead': lead,
            'risk_scale': max(.4, 1. - .6 * pressure + .5 * lead),
            'upside': MAX_UPSIDE * pressure,
            'explore': MAX_EXPLORE * pressure * (1. - lead)}


def note_cash(history, state):
    """Record every truck's cash for today (day -> {id: cash}); keep a short window."""
    try:
        day = int(state.get('day'))
        cash = {t['id']: float(t['cash']) for t in state.get('trucks') or [] if t.get('id') and t.get('cash') is not None}
        mine = state.get('you') or {}
        if mine.get('id') and mine.get('cash') is not None:
            cash[mine['id']] = float(mine['cash'])
        if cash:
            history[day] = cash
            for old in sorted(history)[:-(WINDOW + 3)]:
                del history[old]
    except (TypeError, ValueError, KeyError):
        pass
    return history
