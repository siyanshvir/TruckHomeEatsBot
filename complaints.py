"""Health complaints against a rival: a rarely used move, filed only when the numbers clearly favor it.

The game's tip line cuts the target's reputation 20% at every spot from the next day. Each complaint can
be caught: with k complaints in the last 3 days (this one included) the catch chance is 10% x k, and a
catch bans us (5 days live; the rest of the market in the final), throws the complaint out and restores
the targets. So the model weighs the gain against catch risk x loss:

    gain  = our extra expected profit tomorrow if the target's reputation were 20% lower, times how long
            the dent lasts (reputation recovers as the target sells)
    loss  = days we would miss x our recent daily profit (more days left, more to lose)
    file  = gain - p_catch * (loss + gain) clears a margin, at most one complaint per 3 days, and only
            against a rival that is likely to share our spot.

Because loss shrinks as the market ends, filing becomes attractive only late. Everything is switchable:
NIGHTSHIFT_COMPLAINTS=off or data/profile.json {"complaints": "off"}.
"""
import json
import os
from pathlib import Path

ENDPOINT = '/x/8243598d/complaints'
WINDOW_DAYS = 3
CATCH_STEP = .10
LIVE_BAN_DAYS = 5
PERSISTENCE = 2.0        # a 20% dent lasts about this many days of full effect as reputation recovers
OVERLAP_MIN = .35        # the target must likely share our spot
MIN_GAIN = 40.
MARGIN = 25.


def enabled(mode, root=None):
    """On in the final by default; elsewhere only when asked for."""
    flag = os.environ.get('NIGHTSHIFT_COMPLAINTS')
    if flag in ('on', 'off'):
        return flag == 'on'
    try:
        path = Path(root) if root else Path(__file__).resolve().parent / 'data' / 'profile.json'
        value = json.loads(path.read_text()).get('complaints')
        if value in ('on', 'off'):
            return value == 'on'
    except (OSError, ValueError, AttributeError):
        pass
    return mode in ('final', 'showcase')


def catch_chance(filed_recently):
    return min(.95, CATCH_STEP * (filed_recently + 1))


def ban_days(state):
    """Days we would miss if caught: the rest of the market in the final, 5 in the live game."""
    left = max(0, int(state.get('days') or 0) - int(state.get('order_for_day') or state.get('day') or 0) + 1)
    return left if state.get('mode') in ('final', 'showcase', 'practice') else min(LIVE_BAN_DAYS, left)


class Complaints:
    def __init__(self, data=None):
        data = data or {}
        self.filed = [tuple(x) for x in data.get('filed', [])]     # (market key, day, target)
        self.caught = int(data.get('caught', 0))
        self.log = data.get('log', [])[-20:]

    def to_dict(self):
        return {'filed': self.filed[-30:], 'caught': self.caught, 'log': self.log[-20:]}

    def recent(self, market, day):
        return sum(1 for m, d, _ in self.filed if m == market and day - d < WINDOW_DAYS)

    def analyse(self, state, brain, order, daily_profit):
        """Best target and its expected value, or a reason not to file."""
        market, day = tuple((state.get('mode'), state.get('market'))), int(state.get('order_for_day') or 0)
        market = list(market)
        if self.recent(market, day):
            return {'file': False, 'reason': 'one complaint per %d days' % WINDOW_DAYS}
        if not order or not order.get('meals'):
            return {'file': False, 'reason': 'no order'}
        me = (state.get('you') or {}).get('id')
        rivals = [t['id'] for t in state.get('trucks') or [] if t.get('id') and t['id'] != me]
        if not rivals:
            return {'file': False, 'reason': 'no rivals'}
        p_catch = catch_chance(0)
        loss = ban_days(state) * max(0., daily_profit)
        base = brain.order_value(state, order, {})
        overlaps = dict(brain.last_overlap)
        likely = sorted((t for t in rivals if overlaps.get(t, 0.) >= OVERLAP_MIN), key=lambda t: -overlaps[t])[:3]
        best = None
        for target in likely:
            hit = brain.order_value(state, order, {target: .8})
            overlap = overlaps.get(target, 0.)
            gain = (hit['profit'] - base['profit']) * PERSISTENCE
            ev = gain - p_catch * (loss + gain)
            row = {'target': target, 'gain': gain, 'overlap': overlap, 'ev': ev, 'loss': loss, 'p_catch': p_catch}
            if overlap >= OVERLAP_MIN and (best is None or ev > best['ev']):
                best = row
        if best is None:
            return {'file': False, 'reason': 'no rival likely to share our spot'}
        best['file'] = best['gain'] >= MIN_GAIN and best['ev'] >= max(MARGIN, .25 * best['gain'])
        best['reason'] = 'gain $%.0f, catch risk x loss $%.0f, ev $%.0f' % (best['gain'], p_catch * (loss + best['gain']), best['ev'])
        return best

    def record(self, state, target, response):
        market = list(tuple((state.get('mode'), state.get('market'))))
        self.filed.append((market, int(state.get('order_for_day') or 0), target))
        caught = isinstance(response, dict) and any(
            bool(v) for k, v in response.items() if 'caught' in str(k).lower() or 'banned' in str(k).lower())
        self.caught += bool(caught)
        self.log = (self.log + [{'day': state.get('order_for_day'), 'target': target, 'caught': bool(caught),
                                 'response': str(response)[:200]}])[-20:]
        return caught
