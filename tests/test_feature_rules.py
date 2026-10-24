import math
import unittest
from game import expected_sales, expected_market_sales, WALK_DISTANCES


MODEL = (100., 1., (1., 0., 0.))


class FeatureRuleTests(unittest.TestCase):
    def test_default_retains_legacy_tie_priority(self):
        trucks = [(8., 'premium', 50., 100), (8., 'premium', 50., 100)]
        self.assertEqual(expected_sales(trucks, 50, MODEL), [50., 0.])
        self.assertEqual(expected_sales(trucks, 50, MODEL, fair_choice=False), [50., 0.])
        market = expected_market_sales({'oval': trucks}, {'oval': 50}, {'oval': MODEL})
        self.assertEqual(market['oval'], [50., 0.])

    def test_fair_equal_trucks_share_and_utility_softmax_ratio(self):
        trucks = [(8., 'premium', 50., 100), (8., 'premium', 50., 100)]
        self.assertEqual(expected_sales(trucks, 50, MODEL, fair_choice=True), [25., 25.])
        trucks[1] = (9.5, 'premium', 50., 100)
        sales = expected_sales(trucks, 50, MODEL, fair_choice=True)
        self.assertAlmostEqual(sales[0]/sales[1], math.e)
        self.assertAlmostEqual(sum(sales), 50.)

    def test_fair_stockout_redistributes_and_respects_eligibility(self):
        trucks = [(8., 'premium', 50., 5), (8., 'premium', 50., 100)]
        sales = expected_sales(trucks, 50, MODEL, fair_choice=True)
        self.assertAlmostEqual(sales[0], 5.)
        self.assertAlmostEqual(sales[1], 45.)
        limited = (8., .3, (0., 0., 1.))
        trucks = [(3., 'cheap', 50., 100), (30., 'premium', 50., 100), (8., 'premium', 50., 100)]
        sales = expected_sales(trucks, 100, limited, fair_choice=True)
        self.assertEqual(sales[:2], [0., 0.])
        self.assertAlmostEqual(sales[2], 50.)

    def test_near_walk_over_exceeds_far_and_skips_ineligible_nearest(self):
        def run(destination, quality='premium'):
            placements = {'oval': [], destination: [(8., quality, 50., 1000)]}
            return expected_market_sales(placements, {'oval': 100, destination: 0},
                                         {'oval': MODEL, destination: MODEL}, walk_over=True)[destination][0]
        self.assertAlmostEqual(run('thompson'), 100*.7*math.exp(-130/400.))
        self.assertGreater(run('thompson'), run('stadium'))
        placements = {'oval': [], 'thompson': [(3., 'cheap', 50., 100)],
                      'union': [(8., 'premium', 50., 100)]}
        premium = (100., 1., (0., 0., 1.))
        result = expected_market_sales(placements, {'oval': 100, 'thompson': 0, 'union': 0},
                                      dict.fromkeys(placements, premium), walk_over=True)
        self.assertEqual(result['thompson'], [0.])
        self.assertAlmostEqual(result['union'][0], 100*.7*math.exp(-470/400.))

    def test_walking_preserves_original_budget_and_local_stock(self):
        placements = {'oval': [(8., 'premium', 50., 10)],
                      'thompson': [(8., 'premium', 50., 15)],
                      'union': [(9., 'premium', 50., 100)]}
        models = dict.fromkeys(placements, (8., .3, (1., 0., 0.)))
        result = expected_market_sales(placements, {'oval': 100, 'thompson': 20, 'union': 0},
                                      models, fair_choice=True, walk_over=True)
        self.assertLessEqual(sum(sum(s) for s in result.values()), 120.)
        self.assertLessEqual(result['oval'][0], 10.)
        self.assertLessEqual(result['thompson'][0], 15.)
        # Most residual buyers cannot afford $9; walking may not redraw their budget.
        self.assertLess(result['union'][0], 1.)

    def test_no_eligible_destination_no_walk_and_zero_local_crowds(self):
        placements = {'oval': [], 'thompson': [(30., 'premium', 50., 100)]}
        result = expected_market_sales(placements, {'oval': 100, 'thompson': 0},
                                      dict.fromkeys(placements, MODEL), fair_choice=True, walk_over=True)
        self.assertEqual(result, {'oval': [], 'thompson': [0.]})
        self.assertEqual(WALK_DISTANCES['thompson']['oval'], 130)
        self.assertEqual(WALK_DISTANCES['oval']['thompson'], 130)

    def test_multiple_origins_cannot_oversell_destination(self):
        placements = {'oval': [], 'union': [], 'thompson': [(8., 'premium', 50., 7)]}
        result = expected_market_sales(placements, {'oval': 100, 'union': 100, 'thompson': 3},
                                      dict.fromkeys(placements, MODEL), fair_choice=True, walk_over=True)
        self.assertAlmostEqual(result['thompson'][0], 7.)


if __name__ == '__main__':
    unittest.main()
