import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.review_day import review_day


def fixtures():
    order = {'spot': 'northres', 'quality': 'premium', 'price': 7.88, 'meals': 84}
    pre = {'mode': 'live', 'market': 1, 'order_for_day': 14,
           'you': {'id': 'us', 'reputation': {'northres': 50}},
           'trucks': [{'id': 'rival', 'reputation': {'northres': 81}}],
           'forecast': {'expected_customers': {'northres': 101}}}
    decision = {'state': pre, 'order': order,
                'decision': {'chosen': {'order': order, 'sales': 76.74, 'profit': 142.73,
                                        'demand': 85.99, 'base_meals': 82},
                             'rivals': {'rival': {'attendance': .92, 'locations': {'northres': .45}}}}}
    own = dict(order, sold=15, profit=-343.8)
    result = {'mode': 'live', 'market': 1,
              'last_day': {'day': 14, 'you': own,
                           'spots': {'northres': {'customers': 83, 'hungry': 9,
                                                 'trucks': [dict(order, id='us', sold=15),
                                                            {'id': 'rival', 'truck_name': 'Camper',
                                                             'quality': 'premium', 'price': 9.65,
                                                             'meals': 60, 'sold': 59}]}}}}
    return decision, result


class DayReviewTests(unittest.TestCase):
    def write(self, root, decisions, results):
        root = Path(root)
        (root/'decisions-v1.jsonl').write_text('\n'.join(json.dumps(r) for r in decisions))
        (root/'live-observations').mkdir(exist_ok=True)
        for index, row in enumerate(results):
            (root/'live-observations'/('%d.json' % index)).write_text(json.dumps(row))

    def test_day14_accounting_uses_pre_day_reputations(self):
        decision, result = fixtures()
        with tempfile.TemporaryDirectory() as root:
            self.write(root, [decision], [result])
            report = review_day(root, 14)
        actual = report['actual']
        self.assertAlmostEqual(actual['revenue'], 118.2)
        self.assertEqual(actual['ingredient_cost'], 462)
        self.assertAlmostEqual(actual['profit'], -343.8)
        self.assertEqual(actual['unsold'], 69)
        self.assertEqual(actual['unsold_ingredient_cost'], 379.5)
        self.assertEqual(actual['break_even_sales'], 59)
        self.assertAlmostEqual(report['prediction']['sales_error_actual_minus_expected'], -61.74)
        rival = report['actual_rivals'][0]
        self.assertEqual(rival['pre_day_reputation'], 81)
        self.assertAlmostEqual(rival['value_minus_price'], 9.326)
        self.assertAlmostEqual(rival['value_advantage_over_us'], 1.206)
        self.assertAlmostEqual(rival['predicted_joint_presence'], .414)
        self.assertEqual(report['inventory_cushion']['extra_ingredient_cost'], 11)
        self.assertAlmostEqual(report['inventory_cushion']['fixed_observed_sales_base_profit'], -332.8)

    def test_inventory_hypothetical_caps_sales_at_base_quantity(self):
        decision, result = fixtures()
        decision['decision']['chosen']['base_meals'] = 10
        with tempfile.TemporaryDirectory() as root:
            self.write(root, [decision], [result])
            report = review_day(root)
        self.assertAlmostEqual(report['inventory_cushion']['fixed_observed_sales_base_profit'], 23.8)

    def test_missing_decision_keeps_actual_accounting_without_invented_reputation(self):
        _, result = fixtures()
        with tempfile.TemporaryDirectory() as root:
            self.write(root, [], [result])
            report = review_day(root)
        self.assertFalse(report['decision_available'])
        self.assertEqual(report['day'], 14)
        self.assertEqual(report['actual']['sold'], 15)
        self.assertNotIn('actual_rivals', report)
        self.assertTrue(report['warnings'])

    def test_missing_result_and_empty_directory_are_reported(self):
        decision, _ = fixtures()
        with tempfile.TemporaryDirectory() as root:
            empty = review_day(root)
            self.write(root, [decision], [])
            pending = review_day(root)
        self.assertIsNone(empty['day'])
        self.assertFalse(empty['result_available'])
        self.assertEqual(pending['day'], 14)
        self.assertTrue(pending['decision_available'])
        self.assertFalse(pending['result_available'])

    def test_matching_requires_mode_market_and_day_and_excludes_different_order(self):
        decision, result = fixtures()
        wrong = copy.deepcopy(decision)
        wrong['state']['market'] = 2
        wrong['decision']['chosen']['profit'] = 999
        result['last_day']['you']['meals'] = 80
        with tempfile.TemporaryDirectory() as root:
            self.write(root, [decision, wrong], [result])
            report = review_day(root, 14)
        self.assertFalse(report['order_matches'])
        self.assertEqual(report['prediction']['profit'], 142.73)
        self.assertIsNone(report['prediction']['profit_error_actual_minus_expected'])
        self.assertNotIn('inventory_cushion', report)

    def test_actual_salvage_payout_is_authoritative_and_profit_comparison_includes_it(self):
        decision, result = fixtures()
        result['last_day']['you'].update(salvage=94.875, profit=-248.925)
        with tempfile.TemporaryDirectory() as root:
            self.write(root, [decision], [result])
            report = review_day(root)
        actual = report['actual']
        self.assertAlmostEqual(actual['profit'], -248.925)
        self.assertAlmostEqual(actual['profit_difference_from_reported'], 0.)
        self.assertEqual(actual['unsold_ingredient_gross_cost'], 379.5)
        self.assertAlmostEqual(actual['unsold_ingredient_net_cost'], 284.625)
        self.assertEqual(actual['break_even_sales'], 54)
        self.assertAlmostEqual(report['inventory_cushion']['fixed_observed_sales_base_salvage'], 92.125)
        self.assertAlmostEqual(report['inventory_cushion']['fixed_observed_sales_base_profit'], -240.675)
        self.assertAlmostEqual(report['prediction']['profit_error_actual_minus_expected'], -391.655)

    def test_realized_flags_override_captured_flags_and_ignore_future_result_flags(self):
        decision, result = fixtures()
        decision['state']['features'] = {'salvage': True}
        result['last_day']['features'] = {'salvage': False}
        result['features'] = {'salvage': True}
        with tempfile.TemporaryDirectory() as root:
            self.write(root, [decision], [result])
            report = review_day(root)
        self.assertEqual(report['actual']['salvage'], 0.)
        self.assertEqual(report['actual']['salvage_feature_source'], 'last_day.features')
        self.assertAlmostEqual(report['actual']['profit'], -343.8)
        self.assertEqual(report['inventory_cushion']['fixed_observed_sales_base_salvage'], 0.)

    def test_captured_flags_allow_derived_salvage_when_payout_missing(self):
        decision, result = fixtures()
        decision['state']['features'] = {'salvage': {'enabled': True}}
        with tempfile.TemporaryDirectory() as root:
            self.write(root, [decision], [result])
            report = review_day(root)
        self.assertEqual(report['actual']['salvage'], 94.875)
        self.assertEqual(report['actual']['salvage_feature_source'], 'accepted decision state.features')
        self.assertAlmostEqual(report['actual']['profit_difference_from_reported'], 94.875)

    def test_actual_zero_salvage_overrides_enabled_flag_for_actual_accounting(self):
        decision, result = fixtures()
        result['last_day']['features'] = {'salvage': True}
        result['last_day']['you']['salvage'] = 0
        with tempfile.TemporaryDirectory() as root:
            self.write(root, [decision], [result])
            report = review_day(root)
        self.assertEqual(report['actual']['salvage'], 0.)
        self.assertAlmostEqual(report['actual']['profit'], -343.8)
        self.assertEqual(report['actual']['unsold_ingredient_net_cost'], 379.5)
        self.assertEqual(report['inventory_cushion']['fixed_observed_sales_base_salvage'], 92.125)

    def test_missing_flags_zero_payout_keeps_legacy_accounting(self):
        decision, result = fixtures()
        result['features'] = {'salvage': True}
        result['last_day']['you']['salvage'] = 0
        with tempfile.TemporaryDirectory() as root:
            self.write(root, [decision], [result])
            report = review_day(root)
        self.assertAlmostEqual(report['actual']['profit'], -343.8)
        self.assertFalse(report['actual']['salvage_enabled_for_hypothetical_inventory'])
        self.assertEqual(report['inventory_cushion']['fixed_observed_sales_base_salvage'], 0.)

    def test_malformed_records_do_not_prevent_review(self):
        decision, result = fixtures()
        with tempfile.TemporaryDirectory() as root:
            self.write(root, [decision], [result])
            with (Path(root)/'decisions-v1.jsonl').open('a') as out:
                out.write('\n{broken')
            (Path(root)/'live-observations'/'bad.json').write_text('{bad')
            report = review_day(root)
        self.assertEqual(report['actual']['sold'], 15)
        self.assertEqual(len(report['warnings']), 2)


if __name__ == '__main__':
    unittest.main()
