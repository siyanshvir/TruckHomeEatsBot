"""Offline hindsight review of a frozen accepted decision and its public result.

Reads local files only. Alternative inventory figures hold observed sales fixed;
this is accounting, not proof that another quantity or location would win.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from game import Rules, identity, reputation_value, spot_results


def _key(state, day):
    return state.get('mode'), state.get('market'), day


def load_records(root):
    root = Path(root)
    decisions, results, warnings = [], [], []
    decision_file = root / 'decisions-v1.jsonl'
    if decision_file.exists():
        for number, line in enumerate(decision_file.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if isinstance(row.get('state'), dict):
                    decisions.append(row)
            except (ValueError, AttributeError):
                warnings.append('Skipped invalid decision line %d' % number)
    for path in sorted((root / 'live-observations').glob('*.json')):
        try:
            state = json.loads(path.read_text())
            if isinstance(state, dict) and isinstance(state.get('last_day'), dict):
                results.append(state)
        except (OSError, ValueError):
            warnings.append('Skipped unreadable result %s' % path.name)
    return decisions, results, warnings


def review_day(root, day=None, mode='live', market=1):
    decisions, results, warnings = load_records(root)
    candidates = [s for s in results if s.get('mode') == mode and s.get('market') == market
                  and isinstance(s['last_day'].get('day'), int)]
    if day is None:
        days = [s['last_day']['day'] for s in candidates]
        days += [r['state'].get('order_for_day') for r in decisions
                 if r['state'].get('mode') == mode and r['state'].get('market') == market
                 and isinstance(r['state'].get('order_for_day'), int)]
        day = max(days) if days else None
    result = next((s for s in reversed(candidates) if s['last_day']['day'] == day), None)
    decision = next((r for r in reversed(decisions)
                     if _key(r['state'], r['state'].get('order_for_day')) == (mode, market, day)), None)
    report = {'mode': mode, 'market': market, 'day': day,
              'provenance': 'hindsight diagnostic using local accepted-decision and public-result files',
              'decision_available': decision is not None, 'result_available': result is not None,
              'limitations': ['Alternative inventory accounting holds observed sales fixed; it does not predict new demand.',
                              'No claim that an alternative location would have won.',
                              'Aggregate historical diagnostics do not reconstruct the original sampled lineups.'],
              'warnings': warnings}
    if not result:
        report['warnings'].append('No matching day result; outcome review is unavailable.')
        return report
    last = result['last_day']
    own = last.get('you') or {}
    if own.get('quality') is None or own.get('price') is None:
        report['warnings'].append('No played order in result; food accounting is unavailable.')
        return report
    rules = Rules.read((decision or {}).get('state') or result)
    quality, price = own['quality'], float(own['price'])
    meals, sold = int(own.get('meals', 0)), float(own.get('sold', 0))
    unit_cost = rules.costs[quality]
    revenue, cost = sold * price, meals * unit_cost
    # Read the realized day's flags first. Current result-state flags may already
    # describe tomorrow's engine, so never use them for historical accounting.
    if isinstance(last.get('features'), dict):
        features, feature_source = last['features'], 'last_day.features'
    elif isinstance(((decision or {}).get('state') or {}).get('features'), dict):
        features, feature_source = decision['state']['features'], 'accepted decision state.features'
    else:
        features, feature_source = {}, 'legacy default (no captured flags)'
    flag = features.get('salvage', False)
    enabled = bool(flag.get('enabled', False)) if isinstance(flag, dict) else bool(flag)
    unsold = max(0., meals-sold)
    gross_waste_cost = unsold * unit_cost
    if own.get('salvage') is not None:
        salvage, salvage_source = float(own['salvage']), 'actual last_day.you.salvage'
    else:
        salvage, salvage_source = (.25 * gross_waste_cost if enabled else 0.), 'derived from captured flags'
    # Positive realized salvage establishes eligibility even without flags.
    salvage_rate = .25 if enabled or salvage > 0 else 0.
    profit = revenue-cost+salvage
    reported_profit = own.get('profit')
    break_even_margin = price - salvage_rate * unit_cost
    break_even = (max(0, math.ceil((cost*(1-salvage_rate)-1e-9)/break_even_margin))
                  if break_even_margin > 0 else (0 if cost == 0 else None))
    report['actual'] = {'spot': own.get('spot'), 'quality': quality, 'price': price,
                        'meals': meals, 'sold': sold, 'revenue': revenue, 'ingredient_cost': cost,
                        'profit': profit, 'reported_profit': reported_profit,
                        'profit_difference_from_reported': profit-float(reported_profit) if reported_profit is not None else None,
                        'salvage': salvage, 'salvage_source': salvage_source,
                        'salvage_feature_source': feature_source,
                        'salvage_enabled_for_hypothetical_inventory': bool(salvage_rate),
                        'unsold': unsold, 'unsold_ingredient_cost': gross_waste_cost,
                        'unsold_ingredient_gross_cost': gross_waste_cost,
                        'unsold_ingredient_net_cost': gross_waste_cost-salvage,
                        'sales_contribution': sold*(price-unit_cost),
                        'break_even_sales': break_even}
    if not decision:
        report['warnings'].append('No frozen accepted decision; forecast and pre-day reputation comparisons unavailable.')
        return report
    state, diagnostics = decision['state'], decision.get('decision') or {}
    chosen = diagnostics.get('chosen') or {}
    order = decision.get('order') or chosen.get('order') or {}
    matches = all(own.get(k) == order.get(k) for k in ('spot', 'quality', 'price', 'meals'))
    report['order_matches'] = matches
    report['prediction'] = {k: chosen.get(k) for k in
                            ('sales', 'profit', 'demand', 'expected_unsold', 'sellout_probability', 'loss')}
    report['prediction']['sales_error_actual_minus_expected'] = sold-chosen['sales'] if matches and 'sales' in chosen else None
    report['prediction']['profit_error_actual_minus_expected'] = profit-chosen['profit'] if matches and 'profit' in chosen else None
    base = chosen.get('base_meals')
    if matches and base is not None and 0 <= base <= meals:
        base_sold = min(sold, base)
        base_salvage = salvage_rate * max(0., base-base_sold) * unit_cost
        report['inventory_cushion'] = {'base_meals': base, 'extra_meals': meals-base,
                                      'extra_ingredient_cost': (meals-base)*unit_cost,
                                      'fixed_observed_sales_base_salvage': base_salvage,
                                      'fixed_observed_sales_base_profit': base_sold*price-base*unit_cost+base_salvage,
                                      'method': 'hypothetical cap: sales=min(actual_sold,base_meals); not a causal demand forecast'}
    me = identity(state.get('you') or {})
    roster = {identity(t): t for t in state.get('trucks') or []}
    roster[me] = state.get('you') or {}
    spot = own.get('spot')
    own_rep = (roster[me].get('reputation') or {}).get(spot)
    own_surplus = reputation_value(quality, own_rep, rules.values)-price if own_rep is not None else None
    report['pre_day_own_reputation'] = own_rep
    report['pre_day_own_value_minus_price'] = own_surplus
    report['actual_rivals'] = []
    info = spot_results(last).get(spot) or {}
    report['actual']['spot_customers'] = info.get('customers')
    report['actual']['forecast_customers'] = (state.get('forecast') or {}).get('expected_customers', {}).get(spot)
    for rival in info.get('trucks') or []:
        name = identity(rival)
        if name == me:
            continue
        rep = (roster.get(name, {}).get('reputation') or {}).get(spot)
        surplus = (reputation_value(rival['quality'], rep, rules.values)-rival['price']
                   if rep is not None and rival.get('quality') in rules.values and rival.get('price') is not None else None)
        prediction = (diagnostics.get('rivals') or {}).get(name) or {}
        attendance = prediction.get('attendance')
        location = (prediction.get('locations') or {}).get(spot)
        report['actual_rivals'].append({'id': name, 'name': rival.get('truck_name'),
                                        'price': rival.get('price'), 'quality': rival.get('quality'),
                                        'meals': rival.get('meals'), 'sold': rival.get('sold'),
                                        'pre_day_reputation': rep, 'value_minus_price': surplus,
                                        'value_advantage_over_us': surplus-own_surplus if surplus is not None and own_surplus is not None else None,
                                        'predicted_attendance': attendance, 'predicted_location_given_attendance': location,
                                        'predicted_joint_presence': attendance*location if attendance is not None and location is not None else None})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(Path(__file__).resolve().parents[1] / 'data'))
    parser.add_argument('--day', type=int)
    parser.add_argument('--mode', default='live')
    parser.add_argument('--market', type=int, default=1)
    parser.add_argument('--output')
    args = parser.parse_args()
    report = review_day(args.data, args.day, args.mode, args.market)
    output = Path(args.output) if args.output else Path(args.data) / 'reviews' / ('day-%s.json' % report['day'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    print('Saved review to %s' % output)


if __name__ == '__main__':
    main()
