"""Election laws as the docs describe them, and a deliberately cautious vote."""

# Laws are matched by name (policy key or the text before the colon). Campus-crowd,
# game-day and meat/veggie laws are not modeled: whether the API forecast already
# includes them is unverified, and food is not live.
PER_MEAL = 0.5
MONEY_RATE = .10
MONEY_FIXED = 20.
MONEY_THRESHOLD = 500.
MIN_VOTE_GAIN = 30.  # $/day; below this, abstaining avoids protests against the winner's voters


def law_terms(policies):
    terms = {'floor': None, 'cap': None, 'per_meal': {}, 'money': 0}
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
        elif 'top' in name:
            terms['money'] = 1   # tax the top: opening pays $20, 10% of revenue above $500 is taxed
        elif 'flat' in name:
            terms['money'] = -1  # flat fee: opening costs $20, 10% of revenue above $500 is a bonus
    return terms


def law_profit(terms, sales, price, quality):
    """Net effect of the laws on one day's profit, given the price customers pay."""
    net = sales * terms['per_meal'].get(quality, 0.)
    if terms['money']:
        excess = max(0., sales * price - MONEY_THRESHOLD)
        net += terms['money'] * (MONEY_FIXED - MONEY_RATE * excess)
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
    best, gain = None, MIN_VOTE_GAIN
    for candidate in candidates:
        terms = law_terms(candidate.get('policies'))
        price = charged_price(terms, order['price'])
        value = sales * (price - order['price']) + law_profit(terms, sales, price, order['quality'])
        if value > gain:
            best, gain = candidate['id'], value
    return best
