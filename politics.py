"""Election laws as the docs describe them, and a deliberately cautious vote."""
import random
from game import COSTS

# Laws use the API's params when present, else the documented name. Crowd multipliers
# are scored for the vote only: whether expected_customers already includes a law in
# force is unverified, so planning leaves them to the learned crowd calibration.
# Game-day and meat/veggie laws are not modeled (food is not live).
PER_MEAL = 0.5
MONEY = {'flat fee': (-20., .1, 500.), 'tax top': (20., -.1, 500.)}  # flat, revenue share, threshold
CROWDS = {'curfew': {'highst': .75, 'northres': .75, 'oval': 1.1, 'thompson': 1.1, 'union': 1.1},
          'quiet core': {'oval': .85, 'thompson': .85, 'union': .85, 'highst': 1.2, 'northres': 1.2}}


def law_terms(policies):
    terms = {'floor': None, 'cap': None, 'per_meal': {}, 'money': None, 'crowd': {}, 'game_crowd': {}, 'riot': 1.}
    for policy in policies or []:
        name = str(policy.get('key') or str(policy.get('text', '')).split(':')[0]).lower().replace('_', ' ')
        params = policy.get('params') or {}
        # Any law may carry spot crowd multipliers, whatever it is called: read what the API says.
        if isinstance(params.get('crowd'), dict):
            terms['crowd'] = {s: float(m) for s, m in params['crowd'].items() if isinstance(m, (int, float))}
        if isinstance(params.get('game_crowd'), dict):
            terms['game_crowd'] = {s: float(m) for s, m in params['game_crowd'].items() if isinstance(m, (int, float))}
            terms['riot'] = float(params['riot']) if isinstance(params.get('riot'), (int, float)) else 1.
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
        elif ('curfew' in name or 'quiet' in name) and not terms['crowd']:
            terms['crowd'] = dict(CROWDS['curfew' if 'curfew' in name else 'quiet core'])
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


TERM_DAYS = 8
PROTEST_DAILY, PROTEST_SHARE = .30, .40  # chance of a protest a day; customers lost when targeted


def win_odds(ids, tally, unvoted, ours, samples=400, turnout=.7, seed=7):
    """Chance each candidate wins given our vote (None abstains); unvoted trucks vote at random.

    Ties are settled by a draw, as in the game."""
    rng = random.Random(seed)
    odds = dict.fromkeys(ids, 0.)
    for _ in range(samples):
        counts = {c: float(tally.get(c, 0)) for c in ids}
        for _n in range(unvoted):
            if rng.random() < turnout:
                counts[rng.choice(ids)] += 1
        if ours in counts:
            counts[ours] += 1
        top = max(counts.values())
        leaders = [c for c in ids if counts[c] == top]
        for c in leaders:
            odds[c] += 1. / len(leaders) / samples
    return odds


def vote_analysis(state, order, game_rate=.25, protest_rate=PROTEST_DAILY):
    """Per candidate: daily law value, win odds with and without our vote, protest cost, net gain."""
    election = (state.get('politics') or {}).get('election') or {}
    candidates = [c for c in election.get('candidates') or [] if isinstance(c, dict) and c.get('id')]
    if not candidates or not order or not order.get('meals'):
        return None
    ids = [c['id'] for c in candidates]
    you = state.get('you') or {}
    sales = .85 * order['meals']
    unit = order['price'] - .25 * COSTS[order['quality']]
    value = {}
    for candidate in candidates:
        terms = law_terms(candidate.get('policies'))
        price = charged_price(terms, order['price'])
        crowd = min(order['meals'], sales * terms['crowd'].get(order.get('spot'), 1.))
        daily = ((crowd - sales) * unit + crowd * (price - order['price'])
                 + law_profit(terms, crowd, price, order['quality']))
        game = terms['game_crowd']
        if game:  # on a game day our crowd changes; other days it does not
            on_game = min(order['meals'], crowd * game.get(order.get('spot'), game.get('other', 1.)))
            daily += game_rate * ((on_game - crowd) * unit
                                  + law_profit(terms, on_game, price, order['quality']) - law_profit(terms, crowd, price, order['quality']))
        value[candidate['id']] = daily
    tally = {c: int(n) for c, n in (election.get('tally') or {}).items()}
    own = you.get('vote') if you.get('vote') in ids else None
    if own:
        tally[own] = max(0, tally.get(own, 0) - 1)
    trucks = state.get('trucks') or []
    unvoted = max(0, len(trucks) - 1 - sum(tally.values())) if trucks else 0
    term = TERM_DAYS
    days, vote_day = state.get('days'), election.get('vote_day')
    if days and vote_day:
        term = max(0, min(TERM_DAYS, int(days) - int(vote_day)))
    abstain = win_odds(ids, tally, unvoted, None)
    cash = {t.get('id'): t.get('cash') or 0. for t in trucks}
    revenue = sales * order['price']
    rows = []
    for candidate in candidates:
        cid = candidate['id']
        voted = win_odds(ids, tally, unvoted, cid)
        gain = term * sum((voted[k] - abstain[k]) * value[k] for k in ids)
        voters = [t for t, v in (election.get('votes') or {}).items() if v == cid and t != you.get('id')]
        richest = all(cash.get(t, 0.) <= (you.get('cash') or 0.) for t in voters)
        target = min(1., (2. if richest else 1.) / (len(voters) + 1))  # protests favor the top earner
        protest = term * protest_rate * target * PROTEST_SHARE * revenue * voted[cid]
        rows.append({'id': cid, 'name': candidate.get('name'), 'daily_value': round(value[cid], 1),
                     'p_win_abstain': round(abstain[cid], 3), 'p_win_vote': round(voted[cid], 3),
                     'gain': round(gain, 1), 'protest_cost': round(protest, 1), 'net': round(gain - protest, 1)})
    margin = max(20., .05 * term * sales * unit * .5)
    return {'rows': rows, 'margin': round(margin, 1), 'term_days': term, 'unvoted': unvoted, 'own': own}


def choose_vote(state, order, game_rate=.25, protest_rate=PROTEST_DAILY):
    """Candidate id whose laws most help our usual order, or None to abstain."""
    analysis = vote_analysis(state, order, game_rate, protest_rate)
    if not analysis:
        return None
    best = max(analysis['rows'], key=lambda r: r['net'])
    return best['id'] if best['net'] > analysis['margin'] else None
