"""A rare, rank-aware coupon drop on the rival we are actually racing.

A drop lowers the target's revenue (20% of its buyers pay half), not ours, and costs $20, $40, $80... in a
market. The final ranks average profit, so hurting a close competitor can protect our place, but only if it
is a real contest: so at most one drop per market, only mid-market or later, only on a rival whose cash is
close to ours (neither out of reach nor far behind), and only when its expected loss clearly beats our cost.
Off elsewhere; switch with NIGHTSHIFT_COUPONS=on|off or data/profile.json {"coupons": "on"|"off"}.
"""
import json
import os
from pathlib import Path

MIN_DAY, MIN_LEFT = 10, 2
CONTEST = (.85, 1.25)       # target cash relative to ours that counts as a contest
MIN_LOSS, WORTH = 50., 3.   # the target must lose at least this much, and WORTH x our cost
RECENT = 5


def enabled(mode, root=None):
    flag = os.environ.get('NIGHTSHIFT_COUPONS')
    if flag in ('on', 'off'):
        return flag == 'on'
    try:
        path = Path(root) if root else Path(__file__).resolve().parent / 'data' / 'profile.json'
        value = json.loads(path.read_text()).get('coupons')
        if value in ('on', 'off'):
            return value == 'on'
    except (OSError, ValueError, AttributeError):
        pass
    return mode in ('final', 'showcase')


def recent_revenue(history, truck):
    rows = [r for r in (history.get(truck) or [])[-RECENT:] if r.get('sold') and r.get('price')]
    return sum(r['sold'] * r['price'] for r in rows) / len(rows) if rows else 0.


def decide(state, history, cost):
    """{'drop': True, 'target': id, ...} or {'drop': False, 'reason': ...}."""
    you = state.get('you') or {}
    me, day, days = you.get('id'), int(state.get('day') or 0), int(state.get('days') or 0)
    if not me:
        return {'drop': False, 'reason': 'not playing'}
    if you.get('coupons_dropped'):
        return {'drop': False, 'reason': 'already dropped this market'}
    if day < MIN_DAY or days - day < MIN_LEFT:
        return {'drop': False, 'reason': 'too early or too late in the market'}
    terms = (state.get('rules') or {}).get('coupons') or {}
    share, off = float(terms.get('share', .2)), float(terms.get('off', .5))
    mine, best = float(you.get('cash') or 0.), None
    for truck in state.get('trucks') or []:
        other = truck.get('id')
        if not other or other == me or truck.get('cash') is None or mine <= 0:
            continue
        ratio = float(truck['cash']) / mine
        loss = share * off * recent_revenue(history, other)
        if CONTEST[0] <= ratio <= CONTEST[1] and loss >= max(MIN_LOSS, WORTH * cost):
            if best is None or loss > best['loss']:
                best = {'drop': True, 'target': other, 'loss': loss, 'cost': cost, 'ratio': ratio}
    return best or {'drop': False, 'reason': 'no close rival worth the cost'}
