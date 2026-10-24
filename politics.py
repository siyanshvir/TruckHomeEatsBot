"""Election laws as the docs describe them, and a deliberately cautious vote."""
from game import COSTS

# Laws use the API's params when present, else the documented name. Crowd multipliers
# are scored for the vote only: whether expected_customers already includes a law in
# force is unverified, so planning leaves them to the learned crowd calibration.
# Game-day and meat/veggie laws are not modeled (food is not live).
PER_MEAL = 0.5
MIN_VOTE_GAIN = 30.  # $/day; below this, abstaining avoids protests against the winner's voters
MONEY = {'flat fee': (-20., .1, 500.), 'tax top': (20., -.1, 500.)}  # flat, revenue share, threshold
CROWDS = {'curfew': {'highst': .75, 'northres': .75, 'oval': 1.1, 'thompson': 1.1, 'union': 1.1},
          'quiet core': {'oval': .85, 'thompson': .85, 'union': .85, 'highst': 1.2, 'northres': 1.2}}


def law_terms(policies):
    terms = {'floor': None, 'cap': None, 'per_meal': {}, 'money': None, 'crowd': {}}
    for policy in policies or []:
        name = str(policy.get('key') or str(policy.get('text', '')).split(':')[0]).lower().replace('_', ' ')
        params = policy.get('params') or {}
        if 'floor' in name:
            terms['floor'] = float(params.get('min_price', 7.5))
        elif 'price' in name and 'cap' in name:
            terms['cap'] = float(params.get('max_price', 12.))
        elif 'luxury' in name:
            terms['per_meal'] = {'premium': -PER_MEAL, 'cheap': PER_MEAL}
        elif 'junk' in name:
            terms['per_meal'] = {'cheap': -PER_MEAL, 'premium': PER_MEAL}
        elif 'top' in name or 'flat' in name:
            flat, share, above = MONEY['tax top' if 'top' in name else 'flat fee']
            terms['money'] = (float(params.get('flat', flat)), float(params.get('revenue_share', share)),
                              float(params.get('above', above)))
        elif 'curfew' in name or 'quiet' in name:
            terms['crowd'] = dict(params.get('crowd') or CROWDS['curfew' if 'curfew' in name else 'quiet core'])
    return terms


def law_profit(terms, sales, price, quality):
    """Net effect of the laws on one day's profit, given the price customers pay."""
    net = sales * terms['per_meal'].get(quality, 0.)
    if terms['money']:
        flat, share, above = terms['money']
        net += flat + share * max(0., sales * price - above)
    return net


def charged_price(terms, price):
    if terms['floor'] is not None:
        price = max(price, terms['floor'])
    if terms['cap'] is not None:
        price = min(price, terms['cap'])
    return price


def choose_vote(state, order):
    """Candidate id whose laws most help our usual order, or None to abstain."""
    election = (state.get('politics') or {}).get('election') or {}
    candidates = [c for c in election.get('candidates') or [] if isinstance(c, dict) and c.get('id')]
    if not candidates or not order or not order.get('meals'):
        return None
    sales = .85 * order['meals']
    unit = order['price'] - .25 * COSTS[order['quality']]  # an extra sale also avoids a salvage refund
    best, gain = None, MIN_VOTE_GAIN
    for candidate in candidates:
        terms = law_terms(candidate.get('policies'))
        price = charged_price(terms, order['price'])
        crowd = min(order['meals'], sales * terms['crowd'].get(order.get('spot'), 1.))
        value = ((crowd - sales) * unit + crowd * (price - order['price'])
                 + law_profit(terms, crowd, price, order['quality']))
        if value > gain:
            best, gain = candidate['id'], value
    return best
