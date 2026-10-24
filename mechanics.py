"""Feature-aware projection and food accounting shared by planning and logging."""
from game import expected_sales, expected_market_sales
from learning import feature_tuple

FEATURE_NAMES = ('fair_choice', 'walk_over', 'salvage')


def features(raw=None):
    return dict(zip(FEATURE_NAMES, feature_tuple(raw)))


def food_profit(sales, meals, price, cost, salvage=False):
    return sales*price-meals*cost + max(0., meals-sales)*cost*(.25 if salvage else 0.)


def demand_levels(state, rules, spot, quality, price, scenario):
    """Uncapped demand at three production levels; walking runs after local sales.

    Like the legacy planner, inventory optimization caps this uncapped flow.
    That is an approximation when our stock changes other trucks' leftovers;
    the independent simulator tests finite inventories directly.
    """
    placement, crowds, models, active = scenario
    flags = features(state.get('features'))
    rep = float((state.get('you') or {}).get('reputation', {}).get(spot, 50))
    trucks = list(placement[spot]) + [(price, quality, rep, float('inf'))]
    forecast = state.get('forecast') or {}
    supplied = spot in (forecast.get('expected_customers') or {})
    roster_basis = forecast.get('population_basis') == 'registered_trucks'
    baseline = max(4, len(state.get('trucks') or []))
    levels = ([1., 1., 1.] if supplied and not roster_basis else
              [max(4., active+x)/baseline for x in (0., .5, 1.)])
    demands = {}
    for scale in set(levels):
        if flags['walk_over']:
            market = dict(placement, **{spot: trucks})
            result = expected_market_sales(market, {s: n*scale for s, n in crowds.items()}, models,
                                           rules.values, fair_choice=flags['fair_choice'], walk_over=True)
            demands[scale] = result[spot][-1]
        else:
            demands[scale] = expected_sales(trucks, crowds[spot]*scale, models[spot], rules.values,
                                           fair_choice=flags['fair_choice'])[-1]
    return [demands[scale] for scale in levels]
