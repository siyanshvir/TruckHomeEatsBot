import os
import tempfile
import unittest
from pathlib import Path
from bot import Runner, make_brain
import complaints
from complaints import Complaints, catch_chance, ban_days, enabled
from final_mode import make_final_brain


class FakeBrain:
    """order_value reports `gain` more profit when a rival's reputation is cut."""
    def __init__(self, gain, overlap=None):
        self.gain, self.last_overlap = gain, overlap or {'r1': .8}
    def order_value(self, state, order, hit=None):
        return {'profit': 200. + (self.gain if hit else 0.)}


def state(day, days=30, mode='final', market=1):
    return {'mode': mode, 'market': market, 'day': day - 1, 'order_for_day': day, 'days': days,
            'you': {'id': 'me'}, 'trucks': [{'id': 'me'}, {'id': 'r1'}, {'id': 'r2'}]}


ORDER = {'spot': 'oval', 'quality': 'premium', 'price': 10., 'meals': 80}


class ModelTests(unittest.TestCase):
    def test_catch_chance_follows_the_published_table(self):
        self.assertEqual([round(catch_chance(k), 2) for k in range(4)], [.1, .2, .3, .4])

    def test_the_loss_from_a_ban_shrinks_as_the_market_ends(self):
        self.assertEqual(ban_days(state(1)), 30)
        self.assertEqual(ban_days(state(29)), 2)
        self.assertEqual(ban_days(state(10, mode='live', days=65)), 5)   # live bans are 5 days

    def test_never_files_early_even_for_a_big_gain(self):
        early = Complaints().analyse(state(3), FakeBrain(gain=300.), ORDER, daily_profit=300.)
        self.assertFalse(early['file'])

    def test_files_late_when_the_gain_clearly_beats_the_risk(self):
        late = Complaints().analyse(state(29), FakeBrain(gain=120.), ORDER, daily_profit=300.)
        self.assertTrue(late['file'], late)
        self.assertEqual(late['target'], 'r1')
        self.assertGreater(late['ev'], 25)

    def test_a_small_gain_is_never_worth_it(self):
        small = Complaints().analyse(state(29), FakeBrain(gain=10.), ORDER, daily_profit=300.)
        self.assertFalse(small['file'])

    def test_only_rivals_likely_to_share_our_spot(self):
        far = Complaints().analyse(state(29), FakeBrain(gain=200., overlap={'r1': .1}), ORDER, daily_profit=300.)
        self.assertFalse(far['file'])

    def test_at_most_one_complaint_in_any_three_days(self):
        c = Complaints()
        c.record(state(26), 'r1', {})
        self.assertFalse(c.analyse(state(27), FakeBrain(gain=300.), ORDER, 300.)['file'])
        self.assertFalse(c.analyse(state(28), FakeBrain(gain=300.), ORDER, 300.)['file'])
        self.assertTrue(c.analyse(state(29), FakeBrain(gain=300.), ORDER, 300.)['file'])
        self.assertFalse(Complaints(c.to_dict()).analyse(state(27), FakeBrain(gain=300.), ORDER, 300.)['file'])

    def test_a_catch_is_recorded(self):
        c = Complaints()
        self.assertTrue(c.record(state(29), 'r1', {'caught': True, 'banned_days': 5}))
        self.assertEqual(c.caught, 1)
        self.assertFalse(c.record(state(30), 'r2', {'ok': True}))

    def test_switch_defaults_to_the_final_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'profile.json'
            self.assertTrue(enabled('final', path))
            self.assertFalse(enabled('live', path))
            path.write_text('{"complaints": "off"}')
            self.assertFalse(enabled('final', path))
            path.write_text('{"complaints": "on"}')
            self.assertTrue(enabled('practice', path))
        os.environ['NIGHTSHIFT_COMPLAINTS'] = 'off'
        try:
            self.assertFalse(enabled('final'))
        finally:
            del os.environ['NIGHTSHIFT_COMPLAINTS']


class RunnerTests(unittest.TestCase):
    def runner(self, caught=False):
        class Client:
            def __init__(self): self.posts = []
            def call(self, method, path, body=None, timeout=30):
                self.posts.append((path, body)); return 200, {'caught': caught}
        runner = Runner(Client(), make_final_brain(samples=6))
        runner.accepted_order = ORDER
        return runner

    def test_files_to_the_right_endpoint_once_per_night_and_only_when_the_model_says_so(self):
        runner = self.runner()
        policy = runner.brain.final_policy
        policy.complaint_decision = lambda st, brain, order: {'file': True, 'target': 'r1', 'reason': 'test'}
        os.environ['NIGHTSHIFT_COMPLAINTS'] = 'on'
        try:
            runner.complain(state(29), '/api/v1')
            runner.complain(state(29), '/api/v1')
            self.assertEqual(runner.client.posts, [('/api/v1/x/8243598d/complaints', {'target': 'r1'})])
            policy.complaint_decision = lambda st, brain, order: {'file': False, 'reason': 'no'}
            runner.complain(state(30), '/api/v1')
            self.assertEqual(len(runner.client.posts), 1)
        finally:
            del os.environ['NIGHTSHIFT_COMPLAINTS']

    def test_nothing_is_filed_in_the_live_game_or_when_off(self):
        runner = self.runner()
        runner.brain.final_policy.complaint_decision = lambda st, brain, order: {'file': True, 'target': 'r1', 'reason': 't'}
        runner.complain(state(29, mode='live'), '/api/v1')
        self.assertEqual(runner.client.posts, [])


class PlannerTests(unittest.TestCase):
    def test_order_value_pairs_scenarios_and_prices_a_reputation_cut(self):
        from sim import Market
        market = Market('complaint-eval', rivals=3, days=6, features={'fair_choice': True, 'walk_over': True, 'salvage': True})
        brain = make_final_brain(samples=12)
        state_ = market.state()
        order = brain.decide(state_)
        for _ in range(3):
            state_ = market.play(order)
            order = brain.decide(state_)
        base = brain.order_value(state_, order)
        again = brain.order_value(state_, order)
        self.assertEqual(base['profit'], again['profit'])        # same seed: no noise between calls
        overlaps = dict(brain.last_overlap)
        cut = brain.order_value(state_, order, {t['id']: .8 for t in state_['trucks'] if t['id'] != state_['you']['id']})
        self.assertGreaterEqual(cut['profit'], base['profit'] - 1e-6)   # weaker rivals never hurt us
        self.assertTrue(all(0 <= v <= 1 for v in overlaps.values()))


if __name__ == '__main__':
    unittest.main()
