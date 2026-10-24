"""Normalized, chronological day records for final-bot training and evaluation.

Reads local files only (never writes except FinalLog). Every record's `features` use only days
strictly before it in the same market, so training rows cannot leak the future.
"""
import json
import os
from pathlib import Path
from game import reputation_value

MODES = ('live', 'practice', 'final')
DEFAULT_REPUTATION = 50.


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _read_lines(path):
    """Yield parsed JSON objects; skip blank, malformed and truncated lines."""
    try:
        text = Path(path).read_text()
    except OSError:
        return
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            yield row


def _market_key(market):
    return (0, market, '') if isinstance(market, int) and not isinstance(market, bool) else (1, 0, str(market))


def _sort_key(record):
    day = record['day']
    return (str(record['mode']), _market_key(record['market']), day if isinstance(day, int) else -1)


def _int_day(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _spots(last):
    raw = last.get('spots')
    if isinstance(raw, list):
        raw = {x.get('spot') or x.get('id'): x for x in raw if isinstance(x, dict)}
    if not isinstance(raw, dict):
        return {}
    out = {}
    for spot, info in raw.items():
        if not isinstance(info, dict):
            continue
        out[spot] = {'customers': info.get('customers') or 0, 'hungry': info.get('hungry') or 0,
                     'walked_in': info.get('walked_in') or 0, 'walked_out': info.get('walked_out') or 0,
                     'trucks': [dict(t) for t in info.get('trucks') or [] if isinstance(t, dict)]}
    return out


def _forecast(raw):
    raw = raw if isinstance(raw, dict) else {}
    return {'expected_customers': dict(raw.get('expected_customers') or {}),
            'weather': dict(raw.get('weather') or {}), 'events': list(raw.get('events') or [])}


def _record(mode, market, last, forecast, me, reputation, features, source):
    day = _int_day(last.get('day'))
    if day is None:
        return None
    you = dict(last.get('you') or {})
    you['id'] = me
    you['reputation'] = dict(reputation or {})
    return {'mode': mode, 'market': market, 'day': day, 'weather': last.get('weather'),
            'events': list(last.get('events') or []) if isinstance(last.get('events'), (list, tuple)) else [],
            'forecast': _forecast(forecast), 'spots': _spots(last), 'you': you,
            'day_features': dict(features or {}), 'features': {}, 'source': source}


def _live(root):
    folder = Path(root) / 'live-observations'
    if not folder.is_dir():
        return
    forecasts = {}
    for row in _read_lines(Path(root) / 'forecasts-v2.jsonl'):
        key = (row.get('mode'), row.get('market'), _int_day(row.get('day')))
        if key[2] is not None and isinstance(row.get('input_forecast'), dict):
            forecasts[key] = row['input_forecast']  # latest wins
    for path in sorted(folder.glob('*.json')):
        state = _read_json(path)
        last = (state or {}).get('last_day')
        if not isinstance(last, dict) or not isinstance(last.get('spots'), (dict, list)):
            continue
        mode, market = state.get('mode', 'live'), state.get('market')
        day = _int_day(last.get('day'))
        you = state.get('you') if isinstance(state.get('you'), dict) else {}
        record = _record(mode, market, last, forecasts.get((mode, market, day)), you.get('id'),
                         you.get('reputation'), state.get('features'), 'live-observations/' + path.name)
        if record:
            yield record


def _collected(root):
    for row in _read_lines(Path(root) / 'collect.jsonl'):
        pre, last = row.get('pre'), row.get('last_day')
        if not isinstance(pre, dict) or not isinstance(last, dict):
            continue
        me = row.get('me')
        trucks = [t for t in pre.get('trucks') or [] if isinstance(t, dict) and t.get('id') == me]
        reputation = (trucks[0].get('rep') or trucks[0].get('reputation')) if trucks else None
        record = _record('practice', str(row.get('seed')), last, pre.get('forecast'), me,
                         reputation, pre.get('features'), 'collect.jsonl')
        if record:
            record['you']['reputation_before'] = dict(reputation or {})
            yield record


def _final(root):
    for row in _read_lines(Path(root) / 'final-days.jsonl'):
        if _int_day(row.get('day')) is None or not isinstance(row.get('spots'), dict):
            continue
        record = _record('final', row.get('market'), row, row.get('forecast'), (row.get('you') or {}).get('id'),
                         (row.get('you') or {}).get('reputation'), row.get('day_features') or row.get('features'),
                         'final-days.jsonl')
        if record:
            yield record


def _prior_features(history):
    """Features from strictly earlier days of one market."""
    if not history:
        return {'prior_days': 0}
    last = history[-1]
    you = last['you']
    profits = [d['you'].get('profit') for d in history if isinstance(d['you'].get('profit'), (int, float))]
    rivals = [t for info in last['spots'].values() for t in info['trucks'] if t.get('id') != you.get('id')]
    crowd = {s: i['customers'] for s, i in last['spots'].items()}
    return {'prior_days': len(history), 'prev_spot': you.get('spot'), 'prev_price': you.get('price'),
            'prev_quality': you.get('quality'), 'prev_profit': you.get('profit'), 'prev_sold': you.get('sold'),
            'mean_profit': sum(profits)/len(profits) if profits else None,
            'prev_rivals': len(rivals),
            'prev_rival_price': sum(t.get('price', 0) for t in rivals)/len(rivals) if rivals else None,
            'prev_busiest_spot': max(crowd, key=crowd.get) if crowd else None}


def iter_days(data_root, include=MODES, with_predictions=True):
    """Yield normalized day records sorted by (mode, market, day). Malformed input is skipped."""
    data_root = Path(data_root)
    records = []
    if 'live' in include:
        records.extend(_live(data_root))
    if 'practice' in include:
        records.extend(_collected(data_root))
    if 'final' in include:
        records.extend(_final(data_root))
    predictions = {}
    if with_predictions:
        for row in _read_lines(data_root / 'evaluation-v2.jsonl'):
            predictions[(row.get('mode'), row.get('market'), _int_day(row.get('day')))] = row
    # De-duplicate (mode, market, day): later sources replace earlier ones.
    unique = {}
    for record in records:
        unique[(record['mode'], str(record['market']), record['day'])] = record
    ordered = sorted(unique.values(), key=_sort_key)
    history, current = [], None
    for record in ordered:
        market = (record['mode'], str(record['market']))
        if market != current:
            history, current = [], market
        record['features'] = _prior_features(history)
        before = history[-1]['you'].get('reputation') if history else None
        record['you'].setdefault('reputation_before', dict(before or {}))
        prediction = predictions.get((record['mode'], record['market'], record['day']))
        if prediction:
            record['prediction'] = prediction
        history.append(record)
        yield record


def value_ceiling(quality, reputation=None, values=None):
    from game import VALUES
    return reputation_value(quality, DEFAULT_REPUTATION if reputation is None else reputation,
                            values or VALUES)


def calibration_rows(items):
    """Rows of predicted vs actual sales. `items` are evaluation-v2 rows or day records carrying a
    `prediction`. price_ratio = price / value ceiling (reputation 50 when unknown)."""
    rows = []
    for item in items:
        row = item.get('prediction') if isinstance(item.get('prediction'), dict) else item
        order = row.get('order') or {}
        try:
            quality, price = order['quality'], float(order['price'])
            expected, actual = float(row['expected_sales']), float(row['actual_sales'])
        except (KeyError, TypeError, ValueError):
            continue
        reputation = None
        if item is not row:
            reputation = (item.get('you', {}).get('reputation_before') or {}).get(order.get('spot'))
        reputation = row.get('reputation', reputation)
        try:
            ceiling = value_ceiling(quality, reputation)
        except (KeyError, TypeError):
            continue
        if ceiling <= 0:
            continue
        meals = order.get('meals')
        sellout = bool(row.get('sellout')) or (isinstance(meals, (int, float)) and meals > 0 and actual >= meals)
        rows.append({'order': dict(order), 'quality': quality, 'price': price, 'value_ceiling': ceiling,
                     'price_ratio': price/ceiling, 'expected_sales': expected, 'actual_sales': actual,
                     'expected_profit': row.get('expected_profit'), 'actual_profit': row.get('actual_profit'),
                     'sellout': sellout, 'features': row.get('features') or {}, 'surprise': row.get('surprise'),
                     'mode': row.get('mode'), 'market': row.get('market'), 'day': row.get('day')})
    return rows


class FinalLog:
    """Append-only JSONL day log, deduplicated by `key` fields; survives a truncated last line."""
    def __init__(self, path, key=('mode', 'market', 'day')):
        self.path, self.key = Path(path), tuple(key)
        self.seen = set()
        for row in _read_lines(self.path):
            self.seen.add(self._id(row))

    def _id(self, record):
        return json.dumps([record.get(k) for k in self.key], sort_keys=True, default=str)

    def append(self, record):
        """Return True when written, False for a duplicate."""
        identity = self._id(record)
        if identity in self.seen:
            return False
        line = json.dumps(record, allow_nan=False, default=str) + '\n'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        prefix = b''
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open('rb') as existing:
                existing.seek(-1, os.SEEK_END)
                if existing.read(1) != b'\n':
                    prefix = b'\n'  # isolate a truncated tail so the new row stays parseable
        fd = os.open(str(self.path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, prefix + line.encode())  # single write: line is never interleaved
            os.fsync(fd)
        finally:
            os.close(fd)
        self.seen.add(identity)
        return True

    def rows(self):
        return list(_read_lines(self.path))
