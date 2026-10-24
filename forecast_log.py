"""Frozen forecast evidence and append-only evaluations; no API or order writes."""
import hashlib
import json
import math
from pathlib import Path
import time
from game import Rules, identity
from mechanics import features, demand_levels
from overheads import Overheads


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_digest():
    root = Path(__file__).resolve().parent
    files = ('bot.py', 'strategy.py', 'learning.py', 'opponents.py', 'game.py',
             'pricing_models.py', 'forecast_log.py', 'history.py', 'mechanics.py', 'vendor/pricewars_strategies.py')
    return digest({name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files})


SOURCE_DIGEST = source_digest()


def quantiles(values):
    values = sorted(values)
    def at(q):
        position = (len(values)-1)*q
        left = int(position)
        right = min(left+1, len(values)-1)
        return values[left] + (values[right]-values[left])*(position-left)
    return {'p10': at(.1), 'p50': at(.5), 'p90': at(.9)} if values else {}


def capture(state, order, scenarios, named_rivals, model, diagnostics=None):
    """Snapshot the actual planning scenarios; never sample again for reporting."""
    rules = Rules.read(state)
    spot, quality, price, meals = (order[k] for k in ('spot', 'quality', 'price', 'meals'))
    flags = features(state.get('features'))
    overheads = Overheads(state, spot, flags['salvage'])
    contribution = min(1., meals*rules.costs[quality]/250.)
    left = 0 if contribution <= .5 else 1
    fraction = contribution*2 if left == 0 else (contribution-.5)*2
    fc = state.get('forecast') or {}
    rows = []
    for index, (placements, crowds, models, active) in enumerate(scenarios):
        demands = demand_levels(state, rules, spot, quality, overheads.charged(price), (placements, crowds, models, active))
        demand = demands[left]*(1-fraction) + demands[left+1]*fraction
        sales = min(meals, demand)
        rows.append({'placements': placements, 'crowds': crowds, 'customer_models': models,
                     'active': active, 'rival_offers': named_rivals[index] if index < len(named_rivals) else [],
                     'demand': demand, 'sales': sales,
                     'profit': overheads.profit(sales, meals, price, quality, rules.costs[quality])})
    profits = [r['profit'] for r in rows]
    worst = sorted(profits)[:max(1, math.ceil(len(profits)*.2))]
    conditional = {}
    names_available = len(named_rivals) == len(rows)
    for truck in state.get('trucks') or []:
        name = identity(truck)
        if name == identity(state.get('you') or {}):
            continue
        if not names_available:
            conditional[name] = {'name': truck.get('truck_name', name),
                                 'sampled_presence': None, 'present': None, 'absent': None}
            continue
        present = [r for r in rows if any(t['id'] == name and t['spot'] == spot for t in r['rival_offers'])]
        absent = [r for r in rows if not any(t['id'] == name and t['spot'] == spot for t in r['rival_offers'])]
        def summary(group):
            return {'samples': len(group), 'mean_sales': sum(r['sales'] for r in group)/len(group),
                    'mean_profit': sum(r['profit'] for r in group)/len(group),
                    'loss_probability': sum(r['profit'] < 0 for r in group)/len(group)} if group else None
        conditional[name] = {'name': truck.get('truck_name', name),
                             'sampled_presence': len(present)/len(rows),
                             'present': summary(present), 'absent': summary(absent)}
    effective_rules = {'budget': rules.budget, 'costs': rules.costs, 'values': rules.values,
                       'spots': list(rules.spots), 'price_min': rules.price_min, 'price_max': rules.price_max,
                       'features': flags}
    rules_digest = digest(effective_rules)
    regime_id = digest([state.get('mode'), state.get('market'), rules_digest])
    record = {'schema_version': 2, 'generated_at': time.time(), 'source_digest': SOURCE_DIGEST,
              'mode': state.get('mode'), 'market': state.get('market'), 'day': state.get('order_for_day'),
              'api_version': state.get('version'), 'order': dict(order), 'input_forecast': fc,
              'features': flags,
              'effective_rules': effective_rules, 'rules_digest': rules_digest, 'regime_id': regime_id,
              'public_reputations': {identity(t): t.get('reputation') for t in state.get('trucks') or []},
              'decision_diagnostics': diagnostics or {},
              'model_digest': digest(model), 'model_before': model,
              'named_rival_scenarios_available': names_available,
              'risk': {'sales': quantiles([r['sales'] for r in rows]), 'profit': quantiles(profits),
                       'loss_probability': sum(p < 0 for p in profits)/len(profits),
                       'worst_20_percent_mean_profit': sum(worst)/len(worst),
                       'crowds': quantiles([r['crowds'][spot] for r in rows])},
              'conditional_rival_risk': conditional, 'scenarios': rows,
              'note': 'Model scenarios, not calibrated guarantees or observed outcomes.'}
    record['forecast_id'] = digest({k: record[k] for k in
                                  ('mode', 'market', 'day', 'api_version', 'order', 'model_digest', 'source_digest', 'rules_digest')})
    # Detach mutable learner/history objects before any subsequent observations.
    return json.loads(json.dumps(record, allow_nan=False))


TAIL_BYTES, TAIL_ROWS = 2_000_000, 8


def append_unique(path, record, field='forecast_id'):
    path = Path(path)
    needs_newline = False
    if path.exists():
        # Duplicates only ever repeat the latest rows, so read just the tail. Parsing the whole ledger on
        # every order grows with the file (about 100 MB over a final) and would eat the next night's window.
        with path.open('rb') as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - TAIL_BYTES))
            contents = handle.read().decode('utf-8', 'ignore')
        needs_newline = bool(contents and not contents.endswith('\n'))
        for line in contents.splitlines()[-TAIL_ROWS:]:
            try:
                existing = json.loads(line)
            except ValueError:
                continue
            if existing.get(field) == record.get(field):
                return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as out:
        if needs_newline:
            out.write('\n')  # Preserve a truncated last record while keeping new rows readable.
        out.write(json.dumps(record, allow_nan=False) + '\n')
    return True
