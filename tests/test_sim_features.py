import unittest
from unittest.mock import patch

from game import SPOTS
from sim import Market


class SimFeaturesTests(unittest.TestCase):
    def fixed_market(self, features, customers=None, rivals=0):
        market = Market('feature-tests', rivals=rivals, days=1, features=features)
        forecast = {'weather': {'clear': 1.}, 'events': [],
                    'expected_customers': dict.fromkeys(SPOTS, 100)}
        market.schedule = [(forecast, 'clear',
                            {s: list((customers or {}).get(s, [])) for s in SPOTS})]
        market.rival_orders = lambda: {}
        return market

    def test_all_features_off_preserves_legacy_outcomes(self):
        market = Market('features-off', rivals=3, days=3)
        explicit = Market('features-off', rivals=3, days=3,
                          features={'fair_choice': False, 'walk_over': False, 'salvage': False})
        order = dict(spot='northres', quality='premium', price=8.5, meals=60)
        for expected_sold, expected_profit in [(60, 180.), (57, 154.5), (48, 78.)]:
            state = market.play(order)
            other = explicit.play(order)
            self.assertEqual(state, other)
            self.assertEqual(state['last_day']['you']['sold'], expected_sold)
            self.assertEqual(state['last_day']['you']['profit'], expected_profit)
            self.assertFalse(any(state['features'].values()))
            self.assertEqual(state['features'], state['last_day']['features'])

    def test_fair_choice_shares_eligible_customers(self):
        market = self.fixed_market({'fair_choice': True}, {'oval': [(20, 0)]*100}, rivals=1)
        market.rival_orders = lambda: {'r0': dict(spot='oval', quality='cheap', price=4.5, meals=200)}
        state = market.play(dict(spot='oval', quality='cheap', price=3.5, meals=200))
        records = {t['id']: t for t in state['last_day']['spots']['oval']['trucks']}
        self.assertGreater(records['r0']['sold'], 0)
        self.assertGreater(records['us']['sold'], records['r0']['sold'])
        self.assertEqual(sum(t['sold'] for t in records.values()), 100)

    def test_walking_once_nearest_eligible_preserves_stock_and_population(self):
        market = self.fixed_market({'walk_over': True}, {'oval': [(20, 2)]*100}, rivals=1)
        market.rival_orders = lambda: {'r0': dict(spot='highst', quality='premium', price=8.5, meals=5)}
        class AlwaysWalk:
            def random(self):
                return 0.
        with patch('sim.random.Random', return_value=AlwaysWalk()):
            state = market.play(dict(spot='thompson', quality='premium', price=8.5, meals=5))
        spots = state['last_day']['spots']
        self.assertEqual(spots['thompson']['walked_in'], 5)
        self.assertEqual(spots['highst']['walked_in'], 5)
        self.assertEqual(spots['oval']['walked_out'], 10)
        self.assertEqual(spots['oval']['hungry'], 90)
        sales = sum(t['sold'] for info in spots.values() for t in info['trucks'])
        self.assertEqual(sales + sum(info['hungry'] for info in spots.values()), 100)
        self.assertEqual(sum(info['walked_in'] for info in spots.values()),
                         sum(info['walked_out'] for info in spots.values()))
        self.assertTrue(all(t['sold'] <= t['meals'] for info in spots.values() for t in info['trucks']))

    def test_locals_have_priority_before_walkers(self):
        market = self.fixed_market({'walk_over': True},
                                   {'oval': [(20, 2)]*10, 'thompson': [(20, 2)]*5})
        state = market.play(dict(spot='thompson', quality='premium', price=8.5, meals=5))
        self.assertEqual(state['last_day']['spots']['thompson']['walked_in'], 0)
        self.assertEqual(state['last_day']['spots']['oval']['walked_out'], 0)
        self.assertEqual(state['last_day']['you']['sold'], 5)

    def test_quality_and_budget_rejection_survive_features(self):
        market = self.fixed_market({'fair_choice': True, 'walk_over': True},
                                   {'oval': [(20, 2)]*10 + [(1, 0)]*10})
        state = market.play(dict(spot='oval', quality='cheap', price=3.5, meals=20))
        self.assertEqual(state['last_day']['you']['sold'], 0)
        self.assertEqual(state['last_day']['spots']['oval']['hungry'], 20)

    def test_salvage_is_quarter_ingredient_cost_of_unsold_meals(self):
        market = self.fixed_market({'salvage': True})
        state = market.play(dict(spot='oval', quality='premium', price=8.5, meals=10))
        own = state['last_day']['you']
        self.assertEqual(own['salvage'], 13.75)
        self.assertEqual(own['cost'], 55.)
        self.assertEqual(own['profit'], -41.25)
        self.assertEqual(state['you']['cash'], -41.25)
        self.assertEqual(own['profit'], own['revenue']-own['cost']+own['salvage'])
        state['features']['salvage'] = False
        self.assertTrue(state['last_day']['features']['salvage'])


if __name__ == '__main__':
    unittest.main()
