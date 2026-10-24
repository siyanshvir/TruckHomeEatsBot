import json
import unittest
from unittest import mock

import game
import rules_adapter
from game import SPOTS, Rules, known_spots, register_spots, reset_spots, weather_factor
from learning import Learner
from sim import Market
from strategy import Brain

NEW = 'medcenter'


def with_new_spot(state, crowd=700., spots=(NEW,)):
    state = json.loads(json.dumps(state))
    state['rules']['spots'] = list(SPOTS) + list(spots)
    info = {s: {'name': s, 'x': 100. * i, 'z': 0., 'region': 'campus', 'opened_day': 1} for i, s in enumerate(SPOTS)}
    for j, s in enumerate(spots):
        info[s] = {'name': s, 'x': 250. + 50. * j, 'z': 90., 'region': 'health', 'opened_day': 4}
        state['forecast']['expected_customers'][s] = crowd
    state['spot_info'] = info
    return state


class NewSpotTests(unittest.TestCase):
    def setUp(self):
        reset_spots()

    def tearDown(self):
        reset_spots()

    def test_registry_extends_in_place_and_resets(self):
        before = (dict(game.PRIORS), {k: dict(v) for k, v in game.DISTANCES.items()}, dict(game.WEATHER))
        register_spots({'mirror': (12., 3., (.4, .4, .2), .1)}, {('mirror', 'oval'): 200.})
        self.assertEqual(known_spots(), SPOTS + ('mirror',))
        self.assertEqual(game.DISTANCES['oval']['mirror'], 200.)
        self.assertEqual(weather_factor('rain', 'mirror'), .5)
        reset_spots()
        self.assertEqual(known_spots(), SPOTS)
        self.assertEqual((game.PRIORS, game.DISTANCES, game.WEATHER), before)

    def test_invalid_registration_changes_nothing(self):
        with self.assertRaises(ValueError):
            register_spots({'lane': (12., 3., (.4, .4), .1)}, {})
        self.assertEqual(known_spots(), SPOTS)

    def test_rules_read_keeps_only_registered_spots(self):
        state = with_new_spot(Market('rr', rivals=3, days=3).state())
        self.assertEqual(Rules.read(state).spots, SPOTS)
        Brain.register_new_spots(state)
        self.assertEqual(Rules.read(state).spots, SPOTS + (NEW,))
        self.assertEqual(weather_factor('rain', NEW), 1.)

    def test_planner_can_choose_best_new_spot(self):
        state = with_new_spot(Market('choose', rivals=3, days=3).state())
        brain = Brain(samples=4)
        order = brain.decide(state)
        self.assertEqual(order['spot'], NEW)
        Rules.read(state).validate(order)
        self.assertIn(NEW, [r['order']['spot'] for r in brain.diagnostics['ranking']])
        rows = brain.forecast_record['scenarios']
        self.assertAlmostEqual(sum(r['profit'] for r in rows) / len(rows), brain.diagnostics['chosen']['profit'])
        self.assertIsNone(brain.diagnostics.get('forecast_capture_error'))

    def test_walk_over_and_other_features_with_new_spot(self):
        base = Market('walk', rivals=3, days=3, features={'fair_choice': True, 'walk_over': True, 'salvage': True}).state()
        state = with_new_spot(base)
        brain = Brain(samples=4)
        order = brain.decide(state)
        Rules.read(state).validate(order)
        rows = brain.forecast_record['scenarios']
        self.assertAlmostEqual(sum(r['profit'] for r in rows) / len(rows), brain.diagnostics['chosen']['profit'])

    def test_failed_hook_equals_ignoring_new_spots(self):
        base = Market('ignore', rivals=3, days=3).state()
        plain = Brain(samples=4)
        expected = plain.decide(base)
        state = with_new_spot(base)
        with mock.patch.object(rules_adapter.RuleAdapter, 'normalize', side_effect=RuntimeError('boom')):
            broken = Brain(samples=4)
            order = broken.decide(state)
        self.assertEqual(known_spots(), SPOTS)
        self.assertEqual(order['spot'] in SPOTS, True)
        self.assertEqual(Rules.read(state).spots, SPOTS)
        self.assertEqual(order, expected)

    def test_baseline_state_never_touches_registry(self):
        state = Market('base', rivals=3, days=3).state()
        with mock.patch.object(rules_adapter.RuleAdapter, 'normalize', side_effect=AssertionError('hook ran')):
            order = Brain(samples=4).decide(state)
        self.assertEqual(known_spots(), SPOTS)
        again = Brain(samples=4).decide(state)
        self.assertEqual(order, again)

    def test_checkpoint_roundtrip_with_new_spot(self):
        state = with_new_spot(Market('ckpt', rivals=3, days=3).state())
        brain = Brain(samples=4)
        brain.decide(state)
        brain.learner._ensure(NEW)
        data = json.loads(json.dumps(brain.dump()))
        self.assertIn(NEW, data['learner']['weights'])
        restored = Brain(samples=4)
        restored.restore(data)
        self.assertIn(NEW, restored.learner.weights)
        Rules.read(state).validate(restored.decide(state))
        # Old checkpoints (six spots only) still load and lazily grow.
        old = Learner().to_dict()
        self.assertNotIn(NEW, old['weights'])
        loaded = Learner.from_dict(old)
        loaded.posterior(NEW)
        self.assertIn(NEW, loaded.weights)

    def test_learning_updates_new_spot_crowd_stats(self):
        state = with_new_spot(Market('learn', rivals=3, days=3).state(), crowd=100.)
        Brain.register_new_spots(state)
        last = {'day': 1, 'weather': 'clear', 'events': [],
                'spots': {NEW: {'customers': 150, 'trucks': [
                    {'id': 'us', 'price': 9., 'quality': 'standard', 'meals': 60, 'sold': 40},
                    {'id': 'r0', 'price': 8., 'quality': 'cheap', 'meals': 60, 'sold': 55}]}}}
        learner = Learner()
        learner.learn(last, {}, state['forecast'], Rules.read(state))
        self.assertEqual(learner.counts[NEW], 1)
        self.assertGreater(learner.crowd[NEW]['clear'][0], 1.)
        self.assertEqual(learner.crowd_variance[NEW]['clear'][0], 1)
        self.assertGreater(learner.crowd_multiplier(NEW, 'clear'), 1.)
        self.assertTrue(0.04 <= learner.crowd_noise(NEW, 'clear') <= .4)
        self.assertEqual(learner.counts['oval'], 0)
        # A spot that was never registered is ignored, as before.
        ignored = Learner()
        ignored.learn(dict(last, spots={'nowhere': last['spots'][NEW]}), {}, None, Rules.read(state))
        self.assertNotIn('nowhere', ignored.weights)

    def test_brain_observe_learns_new_spot_after_play(self):
        state = with_new_spot(Market('observe', rivals=3, days=3).state())
        brain = Brain(samples=4)
        order = brain.decide(state)
        state = json.loads(json.dumps(state))
        state['last_day'] = {'day': 1, 'weather': 'clear', 'events': [],
                             'spots': {order['spot']: {'customers': 90, 'trucks': [
                                 dict(order, id='us', sold=30)]}},
                             'you': dict(order, sold=30, profit=50., spot=order['spot'])}
        state['day'], state['order_for_day'] = 1, 2
        brain.decide(state)
        self.assertEqual(brain.seen, 1)
        self.assertTrue(brain.history.get('us'))


if __name__ == '__main__':
    unittest.main()
