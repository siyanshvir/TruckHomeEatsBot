import unittest
from bot import fallback, make_brain
from game import SPOTS
from sim import Market
from strategy import Brain


def candidate(spot, profit):
    return {'profit': profit, 'score': profit, 'order': {'spot': spot}}


class SmallCrowdRuleTests(unittest.TestCase):
    def setUp(self):
        self.counts = dict(zip(SPOTS, [92, 70, 72, 63, 94, 29]))
        self.brain = Brain(small_crowds=True)

    def test_top_three_excluded_even_if_predicted_profit_is_higher(self):
        options = [candidate(s, 10000 if s in ('highst', 'union', 'thompson') else 100) for s in SPOTS]
        rule = self.brain.location_restriction(options, self.counts, SPOTS)
        self.assertEqual(rule['eligible'], ['stadium', 'northres', 'oval'])
        self.assertIn(rule['best']['order']['spot'], ['stadium', 'northres'])

    def test_third_requires_clear_profit_advantage(self):
        options = [candidate('stadium', 100), candidate('northres', 110), candidate('oval', 120)]
        self.assertEqual(self.brain.location_restriction(options, self.counts, SPOTS)['best']['order']['spot'], 'northres')
        options[-1] = candidate('oval', 140)
        self.assertEqual(self.brain.location_restriction(options, self.counts, SPOTS)['best']['order']['spot'], 'oval')

    def test_events_change_eligibility_using_updated_forecast(self):
        options = [candidate(s, 100) for s in SPOTS]
        first = self.brain.location_restriction(options, self.counts, SPOTS)['eligible']
        changed = dict(self.counts, stadium=200, thompson=20)
        second = self.brain.location_restriction(options, changed, SPOTS)['eligible']
        self.assertNotEqual(first, second)
        self.assertNotIn('stadium', second)
        self.assertIn('thompson', second)

    def test_default_bot_applies_price_caps_and_customer_rule(self):
        brain = make_brain(samples=4)
        state = Market('low-price-rule', rivals=5, days=3).state()
        state['forecast']['expected_customers'] = self.counts
        order = brain.decide(state)
        self.assertIn(order['spot'], ['stadium', 'northres', 'oval'])
        self.assertLessEqual(order['price'], brain.low_price_caps[order['quality']])
        for row in brain.diagnostics['ranking']:
            self.assertLessEqual(row['order']['price'], brain.low_price_caps[row['order']['quality']])
            if row['order']['quality'] == 'premium':
                self.assertGreaterEqual(row['order']['price'], 7.25)
        self.assertEqual(brain.risk, .08)

    def test_fallback_obeys_bottom_two_and_standard_cap(self):
        state = Market('fallback', rivals=5, days=3).state()
        state['forecast']['expected_customers'] = self.counts
        order = fallback(state)
        self.assertIn(order['spot'], ['stadium', 'northres'])
        self.assertLessEqual(order['price'], 7.)


if __name__ == '__main__': unittest.main()
