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


class ReviewFixTests(unittest.TestCase):
    def test_restart_during_the_final_resumes_the_final_checkpoint(self):
        import os, tempfile, time
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            live, final = Path(folder) / 'runtime-v1.json', Path(folder) / 'final-runtime-v1.json'
            live.write_text('{}')
            self.assertEqual(Runner.resume_path(live), live)
            final.write_text('{}')
            os.utime(final, (time.time() + 5, time.time() + 5))
            self.assertEqual(Runner.resume_path(live), final)
            self.assertIsNone(Runner.resume_path(None))

    def test_final_mode_never_writes_the_live_checkpoint_even_with_the_kill_switch(self):
        import os, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            live = Path(folder) / 'runtime-v1.json'
            runner = Runner(object(), make_brain(samples=8), live)
            os.environ['NIGHTSHIFT_PROFILE'] = 'baseline'
            try:
                runner.apply_profile({'mode': 'final'})
            finally:
                del os.environ['NIGHTSHIFT_PROFILE']
            self.assertEqual(runner.checkpoint.name, 'final-runtime-v1.json')
            runner.apply_profile({'mode': 'live'})
            self.assertEqual(runner.checkpoint, live)

    def test_surprise_days_teach_nothing_and_reward_is_not_the_forecast_gap(self):
        policy = FinalPolicy()
        calls = []
        class Spy:
            def observe(self, *args): calls.append(args)
        policy.competition = Spy()
        order = {'spot': 'thompson', 'quality': 'standard', 'price': 9., 'meals': 80}
        check = {'order_matches': True, 'order': order, 'actual_sales': 50., 'actual_profit': 150., 'surprise': True, 'sellout': False}
        policy.observe_day({}, {'day': 3}, {'demand': 90., 'calibration': 1., 'profit': 400.}, check, {'thompson': 60.})
        self.assertEqual(calls, [])
        policy.knob_day = 3
        class Controller:
            rewards = []
            def update(self, knobs, reward): self.rewards.append(reward)
        policy.controller = Controller()
        policy.profit_mean = 200.
        policy.observe_day({}, {'day': 3}, {'demand': 90., 'calibration': 1., 'profit': 400.}, dict(check, surprise=False), {'thompson': 60.})
        self.assertEqual(Controller.rewards, [-50.])  # 150 actual vs a 200 running mean, not vs the 400 forecast

    def test_staged_search_keeps_two_distinct_prices(self):
        brain = make_final_brain(samples=12)
        play(brain, days=2)
        self.assertTrue(brain.diagnostics['candidates'] > 6)


class ShowcaseAndPriorTests(unittest.TestCase):
    def test_a_null_you_waits_instead_of_ordering(self):
        class Client:
            def __init__(self): self.gets = 0; self.posts = 0
            def call(self, method, path, body=None, **kwargs):
                if method == 'POST':
                    self.posts += 1
                    return 403, {'error': 'not playing'}
                self.gets += 1
                if self.gets > 3: raise Stop()
                from tests.test_runner import state
                st = state('showcase'); st['you'] = None
                return 200, st
        from tests.test_runner import Stop
        client = Client()
        with self.assertRaises(Stop):
            Runner(client, make_brain(samples=8)).play()
        self.assertEqual(client.posts, 0)

    def test_showcase_uses_the_final_profile_and_checkpoint(self):
        runner = Runner(object(), make_brain(samples=8))
        self.assertEqual(runner.profile_wanted({'mode': 'showcase'}), 'final')

    def test_calibrator_is_seeded_from_trained_priors(self):
        from final_mode import calibrator_seed
        seed = calibrator_seed({'calibration': {'buckets': {'standard': {'0.70-0.80': {'mean': .6, 'n': 4, 'censored': 0}}}}})
        self.assertIn('standard|2', seed['cells'])
        policy = FinalPolicy(priors={'calibration': {'buckets': {'standard': {'0.70-0.80': {'mean': .6, 'n': 4, 'censored': 0}}}}})
        self.assertLess(policy.calibrator.factor('thompson', 'standard', 8.5, 11.5), 1.)

    def test_protest_rate_is_learned_from_office_days(self):
        brain = make_brain(samples=8)
        self.assertAlmostEqual(brain.protest_rate(), .30)
        brain.protests = {'days': 40, 'protests': 4, 'against_us': 0}
        self.assertLess(brain.protest_rate(), .30)


class FinalStartResetTests(unittest.TestCase):
    def test_reset_clears_the_tuner_and_competition_but_keeps_learning_elsewhere(self):
        brain = make_final_brain(samples=8)
        play(brain, days=4)
        policy = brain.final_policy
        policy.controller.update(type(policy.controller.choose())(1.0, .5, .1), 5.)
        policy.competition.observe({'you': {'id': 'x', 'profit': 10}, 'spots': {}}, {'spot': 'oval', 'quality': 'standard', 'price': 8, 'meals': 50}, {'profit': 100})
        calibration, rates = policy.calibrator.to_dict(), (brain.phantom, brain.surprises, brain.protests)
        learner = brain.learner.to_dict()
        self.assertTrue(policy.reset_for_final())
        self.assertFalse(policy.reset_for_final())           # only once
        self.assertEqual(policy.calibrator.to_dict(), calibration)  # price-response learning is kept
        self.assertEqual((brain.phantom, brain.surprises, brain.protests), rates)
        self.assertEqual(brain.learner.to_dict(), learner)    # customer and crowd learning is kept
        self.assertEqual(policy.to_dict()['competition'], type(policy.competition)().to_dict())
        self.assertTrue(policy.final_started)

    def test_a_restart_during_the_final_does_not_reset_again(self):
        brain = make_final_brain(samples=8)
        brain.final_policy.reset_for_final()
        other = make_final_brain(samples=8)
        other.restore(brain.dump())
        self.assertFalse(other.final_policy.reset_for_final())

    def test_runner_resets_once_on_the_first_final_state(self):
        runner = Runner(object(), make_final_brain(samples=8))
        runner.apply_profile({'mode': 'live'})
        self.assertFalse(runner.brain.final_policy.final_started)
        runner.apply_profile({'mode': 'final'})
        self.assertTrue(runner.brain.final_policy.final_started)

    def test_adapter_riot_cost_follows_the_law_multiplier(self):
        from rules_adapter import RuleAdapter
        state = {'features': {'riots': True}, 'forecast': {'events': [{'name': 'The Game', 'effect': 'a riot may follow a Michigan loss'}]},
                 'rules': {'spots': ['highst', 'thompson']}}
        base = RuleAdapter().spot_cost(state, 'highst', include_residual=False)
        crack = dict(state, politics={'laws': [{'key': 'crackdown', 'params': {'game_crowd': {'other': 1.1}, 'riot': 0}}]})
        tail = dict(state, politics={'laws': [{'key': 'tailgate', 'params': {'game_crowd': {'other': .9}, 'riot': 1.5}}]})
        self.assertEqual(RuleAdapter().spot_cost(crack, 'highst', include_residual=False), 0)
        self.assertGreater(RuleAdapter().spot_cost(tail, 'highst', include_residual=False), base)
