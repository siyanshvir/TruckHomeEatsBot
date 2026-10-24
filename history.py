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
        if state.get('forecast') and state.get('order_for_day'):
            brain.pending = {'day': state['order_for_day'], 'forecast': state['forecast']}
        loaded += 1
    return loaded
