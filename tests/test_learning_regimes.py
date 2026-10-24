import copy
import json
import unittest
from unittest.mock import patch

from game import SPOTS
from learning import Learner, PARTICLES, feature_tuple, train


def outcome():
    return {'day': 1, 'weather': 'clear',
            'spots': {'northres': {'customers': 80, 'walked_in': 12, 'walked_out': 4,
                                  'trucks': [{'id': 'r', 'quality': 'premium',
                                              'price': 8., 'meals': 60, 'sold': 20}]}}}


def forecast():
    return {'weather': {'clear': 1.}, 'expected_customers': {'northres': 100}}


class LearningRegimeTests(unittest.TestCase):
    def test_legacy_version_one_load_and_input_mutation_isolation(self):
        legacy = Learner().to_dict()
        legacy.pop('regimes')
        legacy.pop('active_features')
        legacy.pop('crowd_variance')
        legacy['weights']['northres'][0] = -3.
        legacy['rows'] = 420
        learner = Learner.from_dict(legacy)
        self.assertEqual(learner.active_features, (False, False, False))
        self.assertEqual(learner.rows, 420)
        self.assertEqual(learner.weights['northres'][0], -3.)
        self.assertEqual(learner.crowd_variance['northres'], {})
        legacy['weights']['northres'][0] = -99.
        self.assertEqual(learner.weights['northres'][0], -3.)

    def test_new_feature_regime_starts_broad_and_switchback_preserves_state(self):
        learner = Learner()
        learner.learn(outcome(), forecast=forecast())
        legacy = copy.deepcopy(learner._current_bank())
        self.assertTrue(learner.activate({'fair_choice': True}))
        self.assertEqual(learner.weights['northres'], [0.] * len(PARTICLES))
        self.assertEqual(learner.counts['northres'], 0)
        self.assertEqual(learner.rows, 0)
        self.assertEqual(learner.crowd['northres'], {})
        learner.counts['northres'] = 7
        learner.weights['northres'][0] = -9.
        learner.rows = 3
        self.assertTrue(learner.activate({}))
        self.assertEqual(learner._current_bank(), legacy)
        learner.activate({'fair_choice': True})
        self.assertEqual(learner.counts['northres'], 7)
        self.assertEqual(learner.weights['northres'][0], -9.)
        self.assertEqual(learner.rows, 3)

    def test_repeated_activation_is_idempotent(self):
        learner = Learner()
        learner.activate({'fair_choice': True, 'salvage': True})
        learner.weights['oval'][1] = -4.
        learner.counts['oval'] = 2
        learner.rows = 8
        weights = learner.weights
        self.assertFalse(learner.activate({'salvage': True, 'fair_choice': True}))
        self.assertIs(learner.weights, weights)
        self.assertEqual(learner.rows, 8)
        self.assertEqual(learner.weights['oval'][1], -4.)

    def test_all_banks_roundtrip_without_serialized_or_live_aliases(self):
        learner = Learner()
        learner.weights['oval'][0] = -2.
        learner.rows = 4
        learner.activate({'fair_choice': True})
        learner.weights['oval'][0] = -6.
        learner.rows = 9
        snapshot = learner.to_dict()
        restored = Learner.from_dict(json.loads(json.dumps(snapshot)))
        self.assertEqual(restored.active_features, (True, False, False))
        self.assertEqual(restored.rows, 9)
        restored.activate({})
        self.assertEqual(restored.weights['oval'][0], -2.)
        self.assertEqual(restored.rows, 4)
        snapshot['weights']['oval'][0] = -99.
        snapshot['regimes']['000']['weights']['oval'][0] = -77.
        self.assertEqual(learner.weights['oval'][0], -6.)
        self.assertEqual(restored.weights['oval'][0], -2.)
        self.assertIsNot(snapshot['weights'], snapshot['regimes']['100']['weights'])

    def test_fair_choice_uses_realized_flag_in_likelihood(self):
        calls = []
        def estimate(trucks, customers, model, values, **kwargs):
            calls.append(kwargs)
            return [20.]
        learner = Learner()
        with patch('learning.expected_sales', side_effect=estimate):
            learner.learn(outcome(), features={'fair_choice': True})
        self.assertEqual(learner.counts['northres'], 1)
        self.assertEqual(len(calls), len(PARTICLES))
        self.assertTrue(all(call == {'fair_choice': True} for call in calls))

    def test_walk_over_skips_wrong_local_sales_fit_but_keeps_origin_crowd(self):
        learner = Learner()
        with patch('learning.expected_sales') as model:
            learner.learn(outcome(), forecast=forecast(), features={'walk_over': True})
        model.assert_not_called()
        self.assertEqual(learner.counts['northres'], 0)
        self.assertEqual(learner.weights['northres'], [0.] * len(PARTICLES))
        self.assertEqual(learner.rows, 1)
        self.assertEqual(learner.crowd_variance['northres']['clear'], [1, .8, 0.])
        learner.activate({})
        self.assertEqual(learner.rows, 0)
        self.assertEqual(learner.crowd_variance['northres'], {})

    def test_tomorrow_forecast_flags_never_override_realized_flags(self):
        learner = Learner()
        future = dict(forecast(), features={'fair_choice': True, 'walk_over': True})
        realized = dict(outcome(), features={'fair_choice': True})
        calls = []
        def estimate(trucks, customers, model, values, **kwargs):
            calls.append(kwargs)
            return [20.]
        with patch('learning.expected_sales', side_effect=estimate):
            learner.learn(realized, forecast=future, features={})
        self.assertEqual(learner.active_features, (False, False, False))
        self.assertEqual(learner.counts['northres'], 1)
        self.assertTrue(all(not kwargs for kwargs in calls))

    def test_training_rows_take_features_from_pre_not_future_forecast(self):
        future = dict(forecast(), features={'fair_choice': True, 'walk_over': True})
        rows = [{'pre': {'features': {'salvage': True}, 'forecast': future}, 'last_day': outcome()}]
        learner = train(rows)
        self.assertEqual(learner.active_features, (False, False, True))
        self.assertEqual(learner.counts['northres'], 1)
        self.assertEqual(feature_tuple({'fair_choice': {'enabled': False}, 'walk_over': {'enabled': True}}),
                         (False, True, False))

    def test_invalid_banked_weights_rejected(self):
        snapshot = Learner().to_dict()
        snapshot['regimes']['100'] = copy.deepcopy(snapshot['regimes']['000'])
        snapshot['regimes']['100']['weights'][SPOTS[0]] = [float('nan')] * len(PARTICLES)
        with self.assertRaises(ValueError):
            Learner.from_dict(snapshot)


if __name__ == '__main__':
    unittest.main()
