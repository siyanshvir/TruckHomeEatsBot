"""Replay saved public live observations in order; never submit historical orders."""
import json
from pathlib import Path


def warm_start(brain, root):
    directory = Path(root) / 'data' / 'live-observations'
    if not directory.exists():
        return 0
    states = [json.loads(path.read_text()) for path in directory.glob('*.json')]
    states.sort(key=lambda s: (str(s.get('mode')), str(s.get('market')), (s.get('last_day') or {}).get('day', 0)))
    loaded = 0
    for state in states:
        key = [state.get('mode'), state.get('market')]
        if brain.market is not None and brain.market != key:
            continue  # Never replace a final-market checkpoint with old live data.
        day = (state.get('last_day') or {}).get('day')
        if day is None or (brain.seen is not None and day <= brain.seen):
            continue
        brain.observe(state)
        forecast = state.get('forecast') or {}
        next_day = forecast.get('day') or state.get('order_for_day')
        # Day-phase snapshots also contain tomorrow's forecast, even while orders
        # are closed and order_for_day is null. Preserve it for the next result.
        if next_day is not None and next_day > day:
            brain.pending = {'day': next_day, 'forecast': forecast}
            if 'features' in state:
                brain.pending['features'] = state['features']
        loaded += 1
    return loaded
