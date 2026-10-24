import unittest

from game import COSTS, SPOTS
from sim import Market


class FollowerTests(unittest.TestCase):
    def test_follows_our_previous_spot_and_undercuts(self):
        market = Market('follow', rivals=3, days=4, style='follower')
        first = market.rival_orders()
        busiest = max(market.schedule[0][0]['expected_customers'], key=market.schedule[0][0]['expected_customers'].get)
        self.assertTrue(all(o['spot'] == busiest for n, o in first.items()))
        mine = {'spot': 'northres', 'quality': 'premium', 'price': 12., 'meals': 50}
        market.play(mine)
        orders = market.rival_orders()
        for name, order in orders.items():
            self.assertEqual(order['spot'], 'northres')
            self.assertEqual(order['quality'], 'premium')
            self.assertTrue(12.-3.5-.01 <= order['price'] <= 12.-1.5+.01, order)
            self.assertEqual(order['meals'], 90)

    def test_floor_limits_and_standard_match(self):
        market = Market('follow-floor', rivals=2, days=3, style='follower')
        market.play({'spot': 'oval', 'quality': 'standard', 'price': 4., 'meals': 40})
        for order in market.rival_orders().values():
            self.assertEqual(order['quality'], 'standard')
            self.assertGreaterEqual(order['price'], COSTS['standard']+.5)
            self.assertTrue(1 <= order['price'] <= 30)
        market = Market('follow-cheap', rivals=2, days=3, style='follower')
        market.play({'spot': 'oval', 'quality': 'cheap', 'price': 6., 'meals': 40})
        self.assertTrue(all(o['quality'] == 'standard' for o in market.rival_orders().values()))

    def test_deterministic_given_seed(self):
        def trace():
            market, out = Market('follow-det', rivals=3, days=4, style='follower'), []
            for day in range(3):
                out.append(market.rival_orders())
                market.play({'spot': SPOTS[day], 'quality': 'premium', 'price': 9.+day, 'meals': 60})
            return out
        self.assertEqual(trace(), trace())

    def test_other_styles_do_not_depend_on_follower_code(self):
        # The new style only adds branches: other styles still ignore our last order's spot.
        for style in ('mixed', 'crowd', 'moving', 'undercut', 'brutal'):
            a = Market('same', rivals=4, days=3, style=style)
            b = Market('same', rivals=4, days=3, style=style)
            self.assertEqual(a.rival_orders(), b.rival_orders())
            order = {'spot': 'oval', 'quality': 'premium', 'price': 9., 'meals': 60}
            a.play(order); b.play(order)
            self.assertEqual(a.rival_orders(), b.rival_orders())
        self.assertEqual(Market('same', rivals=3, days=2).rival_orders()['r0']['quality'], 'premium')

    def test_follower_runs_with_features(self):
        market = Market('follow-feat', rivals=3, days=3, style='follower', features={'walk_over': True, 'salvage': True})
        for _ in range(3):
            state = market.play({'spot': 'highst', 'quality': 'premium', 'price': 9., 'meals': 60})
        self.assertEqual(state['phase'], 'over')
