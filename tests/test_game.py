import unittest
from game import Rules, expected_sales


class CustomerChoiceTests(unittest.TestCase):
    def test_quality_and_budget_rejections(self):
        self.assertEqual(expected_sales([(5, 'cheap', 50, 100)], 100, (100, 1, (0, 0, 1))), [0.])
        self.assertLess(expected_sales([(12, 'premium', 50, 100)], 100, (4, .3, (1, 0, 0)))[0], .001)

    def test_negative_value_rejected(self):
        self.assertEqual(expected_sales([(9, 'cheap', 50, 100)], 100, (100, 1, (1, 0, 0))), [0.])

    def test_overflow_reaches_second_choice(self):
        sold = expected_sales([(8, 'premium', 50, 20), (9, 'premium', 50, 100)], 100, (100, 1, (0, 0, 1)))
        self.assertAlmostEqual(sold[0], 20)
        self.assertAlmostEqual(sold[1], 80)

    def test_value_tie_favors_lower_price(self):
        sold = expected_sales([(9, 'premium', 50, 100), (5, 'standard', 50, 100)], 50, (100, 1, (0, 1, 0)))
        self.assertEqual(sold, [0., 50.])

    def test_zero_inventory_and_exact_simultaneous_sellout(self):
        sold = expected_sales([(8, 'premium', 50, 0), (9, 'standard', 50, 10), (13, 'premium', 50, 10)],
                              100, (100, 1, (.5, 0, .5)))
        self.assertEqual(sold, [0., 10., 10.])

    def test_state_costs_and_budget_control_capacity(self):
        rules = Rules.read({'rules': {'budget': 100, 'costs': {'cheap': 2, 'standard': 4, 'premium': 10}}})
        self.assertEqual(rules.capacity('premium'), 10)
        with self.assertRaises(AssertionError):
            rules.validate({'spot': 'oval', 'quality': 'premium', 'price': 9, 'meals': 11})


if __name__ == '__main__': unittest.main()
