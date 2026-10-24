import math
import unittest
from copy import deepcopy

from game import DISTANCES, Rules
from rules_adapter import RuleAdapter


def base(**features):
    return {'order_for_day': 3, 'features': features,
            'rules': {'spots': ['oval', 'thompson', 'union', 'fisher', 'zzz']},
            'forecast': {'expected_customers': {'oval': 100, 'thompson': None, 'fisher': 40}},
            'spot_info': {'oval': {'x': 0, 'z': 0}, 'thompson': {'x': 130, 'z': 0},
                          'fisher': {'x': 0, 'z': 300}},
            'you': {}}


class RuleAdapterTests(unittest.TestCase):
    def test_normalize_no_mutation_and_content(self):
        s = base()
        before = deepcopy(s)
        out = RuleAdapter().normalize(s)
        self.assertEqual(s, before)
        for k in before:
            if k != 'forecast':
                self.assertEqual(out[k], before[k])

    def test_priors_and_distances(self):
        out = RuleAdapter().normalize(base())
        pri = out['spot_priors']
        self.assertEqual(set(pri), {'oval', 'thompson', 'union', 'fisher', 'zzz'})
        self.assertAlmostEqual(pri['fisher']['mu'], 14.)
        self.assertAlmostEqual(sum(pri['fisher']['mix']), 1.)
        self.assertTrue(all(k in pri['zzz'] for k in ('mu', 'sd', 'mix', 'weight')))
        d = out['spot_distances']
        self.assertAlmostEqual(d[('oval', 'thompson')], DISTANCES['oval']['thompson'], delta=2)
        self.assertAlmostEqual(d[('oval', 'fisher')], 300.)
        self.assertEqual(d[('oval', 'union')], DISTANCES['oval']['union'])  # fallback to table

    def test_none_counts_dropped(self):
        out = RuleAdapter().normalize(base())
        self.assertEqual(out['forecast']['expected_customers'], {'oval': 100, 'fisher': 40})

    def test_food(self):
        a = RuleAdapter()
        s = base(food=True)
        s['forecast']['diets'] = {'kottman': {'any': .4, 'meat': .1, 'veggie': .5},
                                  'rpac': {'any': .45, 'meat': .45, 'veggie': .1}}
        self.assertEqual(a.order_extras(s, {'spot': 'kottman'}), {'food': 'veggie'})
        self.assertEqual(a.order_extras(s, {'spot': 'rpac'}), {'food': 'meat'})
        s['last_day'] = {'spots': {'kottman': {'trucks': [{'food': 'veggie'}] * 4}}}
        self.assertEqual(a.order_extras(s, {'spot': 'kottman'}), {'food': 'meat'})
        self.assertEqual(a.order_extras(s, {'spot': 'oval'}), {'food': 'meat'})
        self.assertEqual(a.order_extras(base(food=False), {'spot': 'kottman'}), {})

    def test_riot_cost(self):
        a = RuleAdapter()
        s = base(riots=True)
        s['rules']['spots'] = ['highst', 'oval', 'stadium']
        s['spot_info'] = {}
        s['forecast']['events'] = [{'key': 'riot', 'name': 'Michigan loss riot expected'}]
        near, far = a.spot_cost(s, 'highst'), a.spot_cost(s, 'stadium')
        self.assertGreater(near, 50)
        self.assertLess(far, near / 20)
        s['features']['riots'] = False
        self.assertEqual(a.spot_cost(s, 'highst'), 0.)
        self.assertEqual(a.spot_cost(base(riots=True), 'oval'), 0.)

    def test_residual_learning(self):
        a = RuleAdapter()
        s = base()
        day = {'you': {'profit': 80.}}
        a.observe(day, {'spot': 'oval'}, 100.)
        a.observe(day, {'spot': 'oval'}, 100.)
        self.assertEqual(a.spot_cost(s, 'oval'), 0.)
        a.observe(day, {'spot': 'oval'}, 100.)
        self.assertGreater(a.spot_cost(s, 'oval'), 10)
        self.assertEqual(a.spot_cost(s, 'union'), 0.)
        a.observe({'you': {'profit': 99.}}, {'spot': 'union'}, 100.)
        self.assertEqual(a.spot_cost(s, 'union'), 0.)

    def test_malformed(self):
        a = RuleAdapter({'residual': 'x', 'streak': [1], 'profit_proxy': 'bad'})
        for junk in (None, 5, 'x', [], {'features': 3, 'rules': 'a', 'forecast': [1], 'spot_info': {'a': 'b'}},
                     {'features': {'food': 1, 'riots': 1}, 'forecast': {'events': 5, 'diets': 3}}):
            a.normalize(junk)
            a.order_extras(junk, junk)
            a.spot_cost(junk, None)
            a.observe(junk, junk, junk)
            a.coupon_action(junk)
            a.is_wrecked(junk)
        a.observe({'you': {'profit': 'x'}}, {'spot': 3}, 'y')
        self.assertIsNone(a.coupon_action(base()))

    def test_wrecked(self):
        a = RuleAdapter()
        s = base()
        self.assertFalse(a.is_wrecked(s))
        s['you']['wrecked_until_day'] = 3
        self.assertTrue(a.is_wrecked(s))
        s['order_for_day'] = 4
        self.assertFalse(a.is_wrecked(s))

    def test_roundtrip(self):
        a = RuleAdapter()
        for _ in range(4):
            a.observe({'you': {'profit': 50., 'repairs': 0}}, {'spot': 'oval'}, 100.)
        b = RuleAdapter(a.to_dict())
        self.assertEqual(b.to_dict(), a.to_dict())
        self.assertEqual(b.spot_cost(base(), 'oval'), a.spot_cost(base(), 'oval'))


if __name__ == '__main__':
    unittest.main()


class CouponStrategyTests(unittest.TestCase):
    def test_cost_doubles_with_each_drop_and_we_never_drop(self):
        from rules_adapter import RuleAdapter
        state = {'rules': {'coupons': {'cost': 20}}, 'you': {'coupons_dropped': []}}
        self.assertEqual(RuleAdapter.coupon_cost(state), 20.)
        state['you']['coupons_dropped'] = [{'day': 1}, {'day': 2}]
        self.assertEqual(RuleAdapter.coupon_cost(state), 80.)
        self.assertIsNone(RuleAdapter().coupon_action(state))
        self.assertEqual(RuleAdapter.coupon_cost({}), 20.)     # missing fields: the base price

    def test_the_tuner_is_not_punished_for_coupons_dropped_on_us(self):
        from final_mode import FinalPolicy
        policy = FinalPolicy()
        rewards = []
        class Controller:
            def update(self, knobs, reward): rewards.append(reward)
        policy.controller, policy.knob_day, policy.profit_mean = Controller(), 7, 200.
        order = {'spot': 'oval', 'quality': 'standard', 'price': 9., 'meals': 60}
        check = {'order_matches': True, 'order': order, 'actual_sales': 50., 'actual_profit': 150., 'sellout': False}
        policy.observe_day({}, {'day': 7, 'you': {'coupon_loss': 40.}}, {'demand': 60., 'calibration': 1., 'profit': 200.}, check, {'oval': 60.})
        self.assertEqual(rewards, [-10.])      # 150 + 40 lost to coupons, against a 200 running mean
