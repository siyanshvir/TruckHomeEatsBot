import os
import unittest
from bot import Runner, make_brain
from coupon_play import decide, enabled


def state(day=15, days=30, mine=1000., others=None, dropped=None):
    others = others if others is not None else {'r1': 1000., 'r2': 400.}
    return {'mode': 'final', 'market': 1, 'day': day, 'order_for_day': day + 1, 'days': days,
            'rules': {'coupons': {'cost': 20, 'share': .2, 'off': .5, 'growth': 2}},
            'you': {'id': 'me', 'cash': mine, 'coupons_dropped': dropped or []},
            'trucks': [{'id': 'me', 'cash': mine}] + [{'id': k, 'cash': v} for k, v in others.items()]}


HISTORY = {'r1': [{'sold': 90, 'price': 10.}] * 5, 'r2': [{'sold': 90, 'price': 10.}] * 5}


class DecisionTests(unittest.TestCase):
    def test_drops_on_a_close_rival_when_it_clearly_pays(self):
        d = decide(state(), HISTORY, 20.)
        self.assertTrue(d['drop'])
        self.assertEqual(d['target'], 'r1')                 # r2 is far behind: no contest
        self.assertAlmostEqual(d['loss'], .2 * .5 * 900)

    def test_rare_by_design(self):
        self.assertFalse(decide(state(day=5), HISTORY, 20.)['drop'])                      # too early
        self.assertFalse(decide(state(day=29), HISTORY, 20.)['drop'])                     # too late
        self.assertFalse(decide(state(dropped=[{'day': 12}]), HISTORY, 20.)['drop'])      # one per market
        self.assertFalse(decide(state(others={'r1': 3000.}), HISTORY, 20.)['drop'])       # out of reach
        self.assertFalse(decide(state(others={'r1': 200.}), HISTORY, 20.)['drop'])        # far behind us
        self.assertFalse(decide(state(), HISTORY, 400.)['drop'])                          # cost too high
        self.assertFalse(decide(state(), {}, 20.)['drop'])                                # no revenue data
        self.assertFalse(decide(state(), HISTORY, 20.) is None)

    def test_switch_defaults_to_the_final_only(self):
        self.assertTrue(enabled('final'))
        self.assertFalse(enabled('live'))
        os.environ['NIGHTSHIFT_COUPONS'] = 'off'
        try:
            self.assertFalse(enabled('final'))
        finally:
            del os.environ['NIGHTSHIFT_COUPONS']


class RunnerTests(unittest.TestCase):
    def test_posts_once_to_the_coupon_endpoint(self):
        class Client:
            def __init__(self): self.posts = []
            def call(self, method, path, body=None, timeout=30):
                self.posts.append((path, body)); return 200, {}
        runner = Runner(Client(), make_brain(samples=6))
        runner.brain.use_final(True)
        runner.brain.history = {k: list(v) for k, v in HISTORY.items()}
        runner.accepted_order = {'spot': 'oval', 'quality': 'premium', 'price': 10., 'meals': 80}
        runner.coupon(state(), '/api/v1')
        runner.coupon(state(), '/api/v1')
        self.assertEqual(runner.client.posts, [('/api/v1/coupons', {'target': 'r1'})])
        runner.coupon(dict(state(), mode='live', order_for_day=99), '/api/v1')   # not the final
        self.assertEqual(len(runner.client.posts), 1)


if __name__ == '__main__':
    unittest.main()
