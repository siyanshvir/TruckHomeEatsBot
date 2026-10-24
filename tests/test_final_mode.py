import copy
import unittest
from bot import Runner, make_brain
from final_mode import FinalPolicy, attach_policy, make_final_brain
from game import Rules
from sim import Market
from strategy import Brain


def play(brain, days=8, seed='final-mode', rivals=3, features=None):
    market = Market(seed, rivals=rivals, days=days, features=features or {'fair_choice': True, 'walk_over': True, 'salvage': True})
    state, orders = market.state(), []
    for _ in range(days):
        order = brain.decide(state)
        Rules.read(state).validate(order)
        orders.append(order)
        state = market.play(order)
    return orders, state


class FinalModeTests(unittest.TestCase):
    def test_inactive_policy_leaves_baseline_decisions_identical(self):
        plain = Brain(samples=8, small_crowds=True, risk=.08)
        shadow = Brain(samples=8, small_crowds=True, risk=.08)
        attach_policy(shadow)  # attached but not steering
        self.assertEqual(play(plain)[0], play(shadow)[0])

    def test_final_profile_plays_a_full_market_within_all_rules(self):
        brain = make_final_brain(samples=8)
        orders, state = play(brain, days=10)
        self.assertEqual(len(orders), 10)
        self.assertTrue(brain.final_active and not brain.small_crowds)
        self.assertEqual(state['phase'], 'over')
        self.assertTrue(all(o['meals'] <= Rules.read(state).capacity(o['quality']) for o in orders))

    def test_final_profile_can_choose_any_spot(self):
        brain = make_final_brain(samples=8)
        play(brain, days=3)
        self.assertEqual(brain.diagnostics['location_rule']['enabled'], False)
        self.assertEqual(len(brain.diagnostics['ranking']), 6)

    def test_toggle_restores_baseline_settings(self):
        brain = make_brain(samples=8)
        base = (brain.small_crowds, brain.risk, dict(brain.spot_price_lift), brain.samples)
        brain.use_final(True)
        self.assertFalse(brain.small_crowds)
        brain.use_final(False)
        self.assertEqual((brain.small_crowds, brain.risk, brain.spot_price_lift, brain.samples), base)

    def test_a_failing_component_never_breaks_the_decision(self):
        brain = make_final_brain(samples=8)
        policy = brain.final_policy
        class Boom:
            def __getattr__(self, name):
                raise RuntimeError('boom')
        policy.calibrator = policy.competition = policy.adapter = policy.controller = Boom()
        orders, _ = play(brain, days=3)
        self.assertEqual(len(orders), 3)
        self.assertTrue(policy.errors)

    def test_calibration_lowers_predicted_sales_after_high_price_misses(self):
        brain = make_final_brain(samples=8)
        market = Market('calib', rivals=3, days=3, features={'fair_choice': True, 'walk_over': True, 'salvage': True})
        state = market.state()
        brain.decide(state)
        spot, quality = 'thompson', 'standard'
        rules = Rules.read(state)
        before = brain.final_policy.demand_factor(state, spot, quality, 10.9, 10.9, 50.)
        for _ in range(6):
            brain.final_policy.calibrator.observe({'spot': spot, 'quality': quality, 'price': 10.9, 'value_ceiling': 12.,
                                                   'expected_demand': 110., 'actual_sales': 53., 'meals': 100, 'sellout': False})
        after = brain.final_policy.demand_factor(state, spot, quality, 10.9, 10.9, 50.)
        low = brain.final_policy.demand_factor(state, spot, quality, 6.0, 6.0, 50.)
        self.assertLess(after, before)
        self.assertGreater(low, after)

    def test_checkpoint_roundtrip_keeps_policy_state(self):
        brain = make_final_brain(samples=8)
        play(brain, days=4)
        data = copy.deepcopy(brain.dump())
        other = make_final_brain(samples=8)
        other.restore(data)
        self.assertEqual(other.final_policy.to_dict()['calibration'], brain.final_policy.to_dict()['calibration'])

    def test_profile_selection(self):
        runner = Runner(object(), make_brain(samples=8))
        self.assertEqual(runner.profile_wanted({'mode': 'live'}), 'baseline')
        self.assertEqual(runner.profile_wanted({'mode': 'final'}), 'final')
        runner.force_final = True
        self.assertEqual(runner.profile_wanted({'mode': 'practice'}), 'final')


if __name__ == '__main__':
    unittest.main()
