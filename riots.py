"""Riot exposure from the documented formulas. Used only as a score cost, never as profit.

A Michigan loss starts a riot on High Street (intensity 0.7, radius 260 m); a national title starts
one around the Stadium (0.95, 380 m). A truck that opened d metres away is hit with probability
intensity * exp(-d / radius): wrecked 30% of the time (out the next 2 days), otherwise a repair
bill of 30 + 120 * exp(-d / radius). The forecast warns the night before, but its shape is not
documented, so warnings are read leniently from event text; with no warning the cost is zero.
"""
import math
from game import DISTANCES

# (intensity, radius, centre spot, chance the game ends that way: Ohio State loses to Michigan 45%,
# wins the title 80%)
KINDS = {'michigan': (.7, 260., 'highst', .45), 'title': (.95, 380., 'stadium', .80)}
WRECK_SHARE = .3
OUT_DAYS = 2


def warnings(forecast):
    """Riot kinds the forecast mentions, as (kind, centre) pairs; the spot may be named in the entry."""
    found = []
    entries = list((forecast or {}).get('events') or [])
    for key in ('riot', 'riots'):
        extra = (forecast or {}).get(key)
        if extra:
            entries += extra if isinstance(extra, list) else [extra]
    for entry in entries:
        text = ' '.join(str(v) for v in entry.values()) if isinstance(entry, dict) else str(entry)
        text = text.lower()
        if 'riot' not in text:
            continue
        kind = 'title' if ('title' in text or 'champion' in text) else 'michigan'
        centre = entry.get('spot') if isinstance(entry, dict) and entry.get('spot') else KINDS[kind][2]
        found.append((kind, centre))
    return found


def riot_cost(state, spot, daily_profit=250.):
    """Expected dollars of riot cost for opening at this spot tomorrow."""
    if not (state.get('features') or {}).get('riots'):
        return 0.
    total = 0.
    for kind, centre in warnings(state.get('forecast')):
        intensity, radius, _, chance = KINDS[kind]
        distance = 0. if spot == centre else DISTANCES.get(centre, {}).get(spot)
        if distance is None:
            continue
        decay = math.exp(-distance / radius)
        hit = intensity * decay
        total += chance * hit * (WRECK_SHARE * OUT_DAYS * daily_profit + (1 - WRECK_SHARE) * (30. + 120. * decay))
    return total
