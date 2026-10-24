import json
import unittest
import tempfile
from pathlib import Path
from history import warm_start
from game import expected_sales, _customer_segments, VALUES
from learning import Learner
from strategy import Brain
from sim import Market
from game import Rules, SPOTS


class ForecastingTests(unittest.TestCase):
    def test_api_crowds_not_reduced_again_for_inactive_roster(self):
        state = Market('api-counts', rivals=8, days=3).state()
        state['forecast'].pop('population_basis')
        state['order_for_day'] = state['days']
        placements = {s: [] for s in SPOTS}
        crowds = dict.fromkeys(SPOTS, 39)
        models = {s: (100, 1, (0, 0, 1)) for s in SPOTS}
        brain = Brain(samples=4)
        scenarios = [(placements, crowds, models, 3.)]*4
        supplied = brain.evaluate(state, Rules.read(state), 'stadium', 'premium', 8.5, scenarios)
        self.assertEqual(supplied['order']['meals'], 39)
        self.assertEqual(supplied['demand'], 39)
        state['forecast']['population_basis'] = 'registered_trucks'
        synthetic = brain.evaluate(state, Rules.read(state), 'stadium', 'premium', 8.5, scenarios)
        self.assertLess(synthetic['order']['meals'], 20)

    def test_null_forecast_count_is_treated_as_missing(self):
        from bot import fallback
        state = Market('null-count', rivals=3, days=3).state()
        state['order_for_day'] = 2
        state['forecast']['expected_customers']['stadium'] = None
        order = Brain(samples=4, small_crowds=True).decide(state)
        Rules.read(state).validate(order)
        Rules.read(state).validate(fallback(state))
        self.assertIsNone(state['forecast']['expected_customers']['stadium'])

    def test_daytime_history_replay_retains_next_forecast(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)/'data'/'live-observations'
            directory.mkdir(parents=True)
            for day in (1, 2):
                state = {'mode': 'live', 'market': 1, 'phase': 'day', 'order_for_day': None,
                         'you': {'id': 'us'}, 'trucks': [],
                         'last_day': {'day': day, 'weather': 'clear',
                                      'spots': {'oval': {'customers': 100, 'trucks': []}}},
                         'forecast': {'day': day+1, 'weather': {'clear': 1},
                                      'expected_customers': {'oval': 100}}}
                (directory/('%d.json' % day)).write_text(json.dumps(state))
            brain = Brain()
            self.assertEqual(warm_start(brain, folder), 2)
            self.assertEqual(brain.learner.crowd_variance['oval']['clear'][0], 1)
            self.assertEqual(brain.pending['day'], 3)
            self.assertEqual(warm_start(brain, folder), 0)

    def test_supplied_events_and_weather_are_not_multiplied_twice(self):
        state = Market('double-events', rivals=5, days=3).state()
        state['forecast'].update(events=['game', 'concert', 'midterm'],
                                 weather={'rain': 1}, expected_customers=dict.fromkeys(SPOTS, 100))
        brain = Brain()
        self.assertEqual(brain.forecast(state, Rules.read(state)), dict.fromkeys(SPOTS, 100))

    def inventory(self, crowds, constrained=True):
        state = Market('cushion', rivals=3, days=3).state()
        state['order_for_day'] = state['days']
        state['trucks'] = [{'id': 'us'}]
        placements = {s: [] for s in SPOTS}
        models = {s: (100, 1, (0, 0, 1)) for s in SPOTS}
        scenarios = [(placements, dict.fromkeys(SPOTS, crowd), models, 3.) for crowd in crowds]
        return Brain(samples=4, small_crowds=constrained).evaluate(
            state, Rules.read(state), 'oval', 'premium', 8.5, scenarios)

    def test_small_inventory_cushion_requires_predicted_additional_sales(self):
        baseline = self.inventory([30]*4+[35]*6, constrained=False)
        increased = self.inventory([30]*4+[35]*6)
        self.assertEqual(baseline['order']['meals'], 30)
        self.assertGreater(increased['order']['meals'], 30)
        self.assertLessEqual(increased['order']['meals'], 33)
        self.assertGreater(increased['sales'], baseline['sales'])
        self.assertGreaterEqual(increased['profit'], baseline['profit']-2.)
        self.assertEqual(increased['order']['price'], baseline['order']['price'])

    def test_no_extra_inventory_when_demand_is_exhausted_or_zero(self):
        self.assertEqual(self.inventory([30]*10)['order']['meals'], 30)
        self.assertEqual(self.inventory([0]*10)['order']['meals'], 0)
        self.assertNotIn('inventory_cushion', self.inventory([30]*10))

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


class BadPracticeDataTests(unittest.TestCase):
    def test_text_prices_in_a_practice_day_do_not_break_learning_or_observation(self):
        from game import number, spot_results
        self.assertEqual(number('$10.00'), 10.0)
        self.assertEqual(number('1,234'), 1234.0)
        self.assertEqual(number('n/a'), 'n/a')
        market = Market('bad-data', rivals=3, days=4)
        brain = Brain(samples=6, small_crowds=True)
        state = market.state()
        state = market.play(brain.decide(state))
        for info in state['last_day']['spots'].values():
            for truck in info.get('trucks') or []:
                truck['price'] = '$%.2f' % float(truck['price'])
                truck['meals'] = str(truck['meals'])
        state['last_day']['you']['price'] = '$%.2f' % float(state['last_day']['you']['price'])
        brain.observe(state)                       # must not raise
        brain.decide(state)
        cleaned = spot_results(state['last_day'])
        self.assertTrue(all(isinstance(t['price'], float) for i in cleaned.values() for t in i.get('trucks') or []))
