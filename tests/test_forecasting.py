import json
import unittest
from game import expected_sales, _customer_segments, VALUES
from learning import Learner
from strategy import Brain
from sim import Market


class ForecastingTests(unittest.TestCase):
    def test_variability_learns_without_turning_bias_into_noise(self):
        stable, variable = Learner(), Learner()
        forecast = {'weather': {'clear': 1}, 'expected_customers': {'oval': 100}}
        for day in range(40):
            for learner, count in ((stable, 130), (variable, 60 if day % 2 else 200)):
                learner.learn({'day': day, 'weather': 'clear',
                               'spots': {'oval': {'customers': count, 'trucks': []}}}, forecast=forecast)
        self.assertLess(stable.crowd_noise('oval', 'clear'), .06)
        self.assertGreater(variable.crowd_noise('oval', 'clear'), .25)
        self.assertGreater(stable.crowd_multiplier('oval', 'clear'), 1.)
        restored = Learner.from_dict(json.loads(json.dumps(variable.to_dict())))
        self.assertEqual(restored.crowd_noise('oval', 'clear'), variable.crowd_noise('oval', 'clear'))

    def test_old_checkpoints_keep_prior_uncertainty(self):
        old = Learner().to_dict()
        old.pop('crowd_variance')
        restored = Learner.from_dict(old)
        self.assertAlmostEqual(restored.crowd_noise('oval', 'rain')**2, .15**2/3)

    def test_cached_preferences_preserve_stock_overflow_and_rule_changes(self):
        trucks = [(8, 'premium', 50, 20), (9, 'premium', 50, 100)]
        model = (100, 1, (0, 0, 1))
        _customer_segments.cache_clear()
        self.assertEqual(expected_sales(trucks, 50, model), [20., 30.])
        hits = _customer_segments.cache_info().hits
        self.assertEqual(expected_sales(trucks, 100, model), [20., 80.])
        self.assertGreater(_customer_segments.cache_info().hits, hits)
        self.assertEqual(expected_sales(trucks, 100, model, dict(VALUES, premium=1)), [0., 0.])
        trucks[0] = (8, 'premium', 50, 0)
        self.assertEqual(expected_sales(trucks, 100, model), [0., 100.])

    def test_predictions_survive_restart_and_are_checked_once(self):
        market = Market('checks', rivals=3, days=3)
        brain = Brain(samples=4, small_crowds=True)
        order = brain.decide(market.state())
        restored = Brain(samples=4, small_crowds=True)
        restored.restore(json.loads(json.dumps(brain.dump())))
        state = market.play(order)
        restored.observe(state)
        restored.observe(state)
        self.assertEqual(len(restored.prediction_checks), 1)
        check = restored.prediction_checks[0]
        self.assertTrue(check['order_matches'])
        self.assertEqual(check['actual_profit'], state['last_day']['you']['profit'])
        if check['sellout']:
            self.assertGreaterEqual(check['demand_error'], 0.)

    def test_replaced_order_is_not_used_as_demand_error(self):
        market = Market('replaced', rivals=3, days=3)
        brain = Brain(samples=4)
        order = brain.decide(market.state())
        replacement = dict(order, meals=0)
        if order['meals'] == 0:
            replacement['price'] += .01
        brain.observe(market.play(replacement))
        self.assertFalse(brain.prediction_checks[-1]['order_matches'])
        self.assertIsNone(brain.prediction_checks[-1]['demand_error'])
