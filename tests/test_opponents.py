import copy
import json
from pathlib import Path
import tempfile
import unittest
from game import SPOTS
from opponents import OpponentModel, context
from pricing_models import reference_prices
from strategy import Brain
from history import warm_start


def forecast(event, spot):
    return {'events': [event], 'weather': {'clear': .5, 'rain': .5},
            'expected_customers': {s: 200 if s == spot else 50 for s in SPOTS}}


def record(spot, fc, price=9.):
    return {'spot': spot, 'quality': 'premium', 'price': price, 'meals': 80, 'sold': 70, 'context': context(fc)}


class OpponentTests(unittest.TestCase):
    def test_tomorrows_event_changes_forecast_with_identical_yesterday(self):
        model = OpponentModel()
        concert = forecast('concert', 'union')
        midterms = forecast('midterm', 'thompson')
        hist = []
        for _ in range(8):
            hist += [record('union', concert), record('thompson', midterms)]
        a = model.placement('r', hist, concert, {'r': hist})
        b = model.placement('r', hist, midterms, {'r': hist})
        self.assertGreater(a[SPOTS.index('union')], b[SPOTS.index('union')])
        self.assertGreater(b[SPOTS.index('thompson')], a[SPOTS.index('thompson')])

    def test_older_history_matters_when_latest_record_is_identical(self):
        fc = forecast('none', 'highst')
        latest = record('highst', fc)
        a = [record('union', fc) for _ in range(10)]+[latest]
        b = [record('oval', fc) for _ in range(10)]+[latest]
        model = OpponentModel()
        pa = model.placement('r', a, fc, {'r': a})
        pb = model.placement('r', b, fc, {'r': b})
        self.assertGreater(pa[SPOTS.index('union')], pb[SPOTS.index('union')])

    def test_price_uses_context_and_multiple_days(self):
        a, b = forecast('concert', 'union'), forecast('midterm', 'thompson')
        hist = [record('union', a, 12.), record('thompson', b, 8.)]*8
        model = OpponentModel()
        high = model.pricing('r', hist, a, {'r': hist}, 'premium')[0]
        low = model.pricing('r', hist, b, {'r': hist}, 'premium')[0]
        self.assertGreater(high, low+.5)
        self.assertGreater(low, 8.)  # Not hard-coded to yesterday's price.

    def test_model_learns_event_follower_before_observation(self):
        model, hist = OpponentModel(), []
        for day in range(40):
            fc = forecast('concert', 'union') if day % 2 else forecast('midterm', 'thompson')
            spot = 'union' if day % 2 else 'thompson'
            rec = record(spot, fc)
            model.observe('r', rec, hist, fc, {'r': hist})
            hist.append(rec)
        entry = model.entry('r')
        self.assertLess(entry['log_loss'], entry['repeat_log_loss'])
        fc = forecast('midterm', 'thompson')
        probabilities = model.placement('r', hist, fc, {'r': hist})
        self.assertEqual(SPOTS[probabilities.index(max(probabilities))], 'thompson')

    def test_absences_reduce_attendance_but_allow_return(self):
        model = OpponentModel()
        original = model.attendance('r')
        for _ in range(8): model.observe('r', None, [], None, {})
        absent = model.attendance('r')
        self.assertLess(absent, original)
        self.assertGreater(absent, 0)
        fc = forecast('concert', 'union')
        for _ in range(12): model.observe('r', record('union', fc), [], fc, {})
        self.assertGreater(model.attendance('r'), absent)

    def test_reference_pricewars_strategies_and_reset(self):
        self.assertEqual(reference_prices([9.], 10., 6., 13.), (8.95, 8.9))
        self.assertEqual(reference_prices([5.], 10., 6., 13.), (6., 13.))
        self.assertEqual(reference_prices([], 10., 6., 13.), (10., 13.))

    def test_history_replay_idempotent_and_cannot_replace_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'data'/'live-observations'
            root.mkdir(parents=True)
            state = {'mode': 'live', 'market': 1, 'you': {'id': 'us'}, 'last_day': {'day': 9, 'spots': {}}, 'trucks': []}
            (root/'day9.json').write_text(json.dumps(state))
            brain = Brain()
            self.assertEqual(warm_start(brain, directory), 1)
            self.assertEqual(warm_start(brain, directory), 0)
            final = Brain(); final.market = ['final', 2]; final.seen = 4
            self.assertEqual(warm_start(final, directory), 0)
            self.assertEqual(final.market, ['final', 2])

    def test_old_checkpoint_remains_loadable(self):
        brain = Brain(); data = brain.dump(); del data['opponents']
        restored = Brain(); restored.restore(data)
        self.assertIsInstance(restored.opponents, OpponentModel)


if __name__ == '__main__': unittest.main()
