import random
import unittest
from unittest.mock import patch
from game import SPOTS, Rules
from opponents import context
from tools.shadow_incumbent import ShadowIncumbentBrain, replay_before


class ShadowIncumbentTests(unittest.TestCase):
    def setUp(self):
        self.brain = ShadowIncumbentBrain(samples=32, small_crowds=True)
        self.forecast = {'day': 14, 'weather': {'clear': 1.},
                         'expected_customers': dict(zip(SPOTS, [150, 120, 114, 101, 88, 48]))}
        self.history = [{'spot': 'northres', 'quality': 'premium', 'price': 9.65,
                         'meals': 60, 'sold': 55, 'context': context(self.forecast), 'day': day}
                        for day in (10, 11, 12)]
        self.history.append({'spot': 'union', 'quality': 'standard', 'price': 11.5,
                             'meals': 82, 'sold': 0, 'context': context(self.forecast), 'day': 13})

    def test_offer_conditions_on_location_but_sparse_location_falls_back(self):
        local = self.brain.offer_history(self.history, 'northres')
        self.assertEqual(len(local), 3)
        self.assertTrue(all(r['quality'] == 'premium' for r in local))
        self.assertIs(self.brain.offer_history(self.history, 'union'), self.history)
        self.assertIs(self.brain.offer_history(self.history, 'stadium'), self.history)

    def test_sampled_offer_uses_the_selected_locations_history(self):
        self.brain.history = {'r': self.history}
        state = {'forecast': self.forecast, 'you': {'id': 'us'},
                 'trucks': [{'id': 'us'}, {'id': 'r'}]}
        rules = Rules.read(state)
        location_weights = [float(spot == 'northres') for spot in rules.spots]
        with patch.object(self.brain.opponents, 'attendance', return_value=1.), \
                patch.object(self.brain.opponents, 'placement', return_value=location_weights), \
                patch.object(self.brain.opponents, 'quality', wraps=self.brain.opponents.quality) as quality, \
                patch.object(self.brain.opponents, 'pricing', wraps=self.brain.opponents.pricing) as pricing:
            self.brain.scenarios(state, rules, self.brain.forecast(state, rules), random.Random(7))
        self.assertEqual(quality.call_count, 32)
        for call in quality.call_args_list:
            self.assertEqual(len(call.args[0]), 3)
            self.assertTrue(all(record['spot'] == 'northres' for record in call.args[0]))
        for call in pricing.call_args_list:
            self.assertTrue(all(record['spot'] == 'northres' for record in call.args[1]))

    def test_public_reputation_exact_even_when_checkpoint_is_stale(self):
        self.brain.history = {'r': self.history}
        self.brain.reputations = {'r': dict.fromkeys(SPOTS, 50.)}
        public = {s: 70.123+i for i, s in enumerate(SPOTS)}
        state = {'forecast': self.forecast, 'you': {'id': 'us'},
                 'trucks': [{'id': 'us'}, {'id': 'r', 'reputation': public}]}
        rules = Rules.read(state)
        scenarios = self.brain.scenarios(state, rules, self.brain.forecast(state, rules), random.Random(3))
        offers = [(spot, t) for placements, _, _, _ in scenarios for spot, trucks in placements.items() for t in trucks]
        self.assertTrue(offers)
        self.assertEqual(len(self.brain.scenario_rivals), len(scenarios))
        for named, (placements, _, _, _) in zip(self.brain.scenario_rivals, scenarios):
            for record in named:
                self.assertEqual(record['id'], 'r')
                self.assertIn((record['price'], record['quality'], record['reputation'], record['meals']),
                              placements[record['spot']])
        for spot, truck in offers:
            self.assertEqual(truck[2], public[spot])

    def test_replay_excludes_target_day_and_future(self):
        states = [{'mode': 'live', 'market': 1, 'you': {'id': 'us'}, 'trucks': [],
                   'last_day': {'day': day, 'spots': {}},
                   'forecast': {'day': day+1, 'expected_customers': {}}} for day in (15, 13, 14)]
        replay_before(self.brain, states, 14)
        self.assertEqual(self.brain.seen, 13)
        self.assertEqual(self.brain.learner.rows, 1)
        self.assertEqual(self.brain.pending['day'], 14)

    def test_decision_retains_three_smallest_rule_and_low_price_caps(self):
        state = {'mode': 'live', 'market': 1, 'day': 13, 'order_for_day': 14,
                 'forecast': self.forecast, 'you': {'id': 'us', 'reputation': dict.fromkeys(SPOTS, 50)},
                 'trucks': [{'id': 'us'}]}
        order = self.brain.decide(state)
        self.assertIn(order['spot'], ['stadium', 'thompson', 'northres'])
        self.assertLessEqual(order['price'], self.brain.low_price_caps[order['quality']])
        if order['quality'] == 'premium':
            self.assertGreaterEqual(order['price'], 7.25)


if __name__ == '__main__':
    unittest.main()
