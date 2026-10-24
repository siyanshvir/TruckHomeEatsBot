import io
import unittest
from contextlib import redirect_stdout
from bot import Runner
from staffing import CrewModel
from strategy import Brain


def state(wage=6., target=4, rivals=((11., 3), (10., 4)), day=24):
    return {'mode': 'live', 'market': 1, 'order_for_day': day, 'features': {'staff': True},
            'you': {'id': 'me', 'staff': {'employees': target, 'target': target, 'wage': wage, 'capacity': 50*target}},
            'trucks': [{'id': 'me'}] + [{'id': 'r%d' % i, 'wage': w, 'employees': e} for i, (w, e) in enumerate(rivals)],
            'labor': {'unemployed': 17, 'lowest_asking_wage': 5.64, 'median_asking_wage': 6.68}}


class CrewModelTests(unittest.TestCase):
    def test_plan_covers_the_order_and_trims_idle_wages(self):
        model = CrewModel(); model.observe(state(), 'me')
        plan = model.plan(state(), {'price': 9.35, 'meals': 90}, [80, 85, 90], 'me')
        self.assertGreaterEqual(plan['headcount']*50, 90)
        self.assertLess(plan['headcount'], 4)
        self.assertGreater(plan['prob_enough_staff'], .9)
        self.assertEqual(plan['rivals_exposed'], 2)
        self.assertEqual(model.pending['target'], plan['headcount'])

    def test_small_order_still_keeps_at_least_one_employee(self):
        plan = CrewModel().plan(state(), {'price': 9.35, 'meals': 0}, [], 'me')
        self.assertGreaterEqual(plan['headcount'], 1)
        self.assertGreaterEqual(plan['wage'], 5.)

    def test_poaching_and_fill_are_learned_and_checkpointed(self):
        model = CrewModel(); model.observe(state(), 'me')
        model.plan(state(), {'price': 9.35, 'meals': 90}, [90], 'me')
        before = model.poach['events'] / model.poach['exposures']
        day = state(); day['last_day'] = {'day': 24, 'you': {'employees': 1}, 'labor': [{'from': 'me', 'to': 'r0', 'count': 1, 'wage': 11}]}
        model.observe(day, 'me')
        self.assertGreater(model.poach['events'] / model.poach['exposures'], before)
        self.assertEqual(model.checks[-1]['actual_employees'], 1)
        self.assertIsNone(model.pending)
        copy = CrewModel(model.to_dict())
        self.assertEqual(copy.poach, model.poach)
        self.assertEqual(copy.checks, model.checks)

    def test_old_checkpoint_without_crew_data_loads(self):
        brain = Brain(samples=4)
        data = brain.dump(); data.pop('crew')
        Brain(samples=4).restore(data)

    def test_runner_sets_crew_once_and_reports_it(self):
        class Client:
            def __init__(self): self.posts = []
            def call(self, method, path, body=None, timeout=30):
                self.posts.append((path, body)); return 200, {}
        class StubBrain:
            diagnostics = {}
            def __init__(self): self.crew = CrewModel()
            def crew_plan(self, st, order): return self.crew.plan(st, order, [88], 'me')
        client = Client(); runner = Runner(client, StubBrain())
        runner.accepted_order = {'spot': 'thompson', 'quality': 'premium', 'price': 9.35, 'meals': 90}
        out = io.StringIO()
        with redirect_stdout(out):
            runner.staff(state(), '/api/v1'); runner.staff(state(), '/api/v1')
        self.assertEqual(len(client.posts), 1)
        self.assertEqual(client.posts[0][0], '/api/v1/staff')
        self.assertIn('crew set', out.getvalue())
        self.assertIn('P(enough)', out.getvalue())


if __name__ == '__main__':
    unittest.main()


class SurpriseTests(unittest.TestCase):
    def test_surprise_scenarios_thin_crowds_only_when_live(self):
        import random
        from game import Rules
        from sim import Market
        base = Market('surprise', rivals=3, days=3).state()
        base['order_for_day'] = 2
        counts = dict.fromkeys(Rules.read(base).spots, 100.)
        def crowds(flag):
            st = dict(base, features=dict(base.get('features') or {}, surprises=flag))
            brain = Brain(samples=400, small_crowds=True)
            return [c['thompson'] for _, c, _, _ in brain.scenarios(st, Rules.read(st), counts, random.Random(1))]
        off, on = crowds(False), crowds(True)
        self.assertGreater(min(off), 60)
        self.assertLess(min(on), .75 * min(off))
        self.assertLess(sum(on)/len(on), sum(off)/len(off))
        self.assertLess(sum(c < .8 * min(off) / .85 for c in on) / len(on), .15)

    def test_surprise_spot_is_flagged_and_not_learned(self):
        from sim import Market
        market = Market('surprise-observe', rivals=3, days=3)
        brain = Brain(samples=4, small_crowds=True)
        state = market.state()
        order = brain.decide(state)
        state = market.play(order)
        state['last_day']['moments'] = [{'kind': 'firedrill', 'spot': order['spot']}]
        before = brain.learner.to_dict()['crowd'][order['spot']]
        brain.observe(state)
        self.assertEqual(brain.learner.to_dict()['crowd'][order['spot']], before)
        self.assertTrue(brain.prediction_checks[-1]['surprise'])


class PoolTests(unittest.TestCase):
    def test_planning_learns_the_pool_without_a_prior_observe(self):
        model = CrewModel()
        plan = model.plan(state(), {'price': 8.42, 'meals': 80}, [66, 80], 'me')
        self.assertAlmostEqual(plan['pool_median_ask'], 6.68)
        model.plan(state(), {'price': 8.42, 'meals': 80}, [66, 80], 'me')
        self.assertAlmostEqual(model.pool['median'], 6.68)  # same day is not averaged twice


class PhantomTests(unittest.TestCase):
    def test_rate_starts_low_and_shrinks_with_quiet_days(self):
        brain = Brain(samples=4)
        start = brain.phantom_rate()
        self.assertTrue(.03 <= start <= .1)
        quiet = [{'order_matches': True, 'actual_rivals': 0, 'expected_rivals': .3} for _ in range(20)]
        brain.phantom = Brain.bootstrap_phantom(quiet)
        self.assertLess(brain.phantom_rate(), start)
        busy = [{'order_matches': True, 'actual_rivals': 2, 'expected_rivals': .3} for _ in range(20)]
        brain.phantom = Brain.bootstrap_phantom(busy)
        self.assertGreater(brain.phantom_rate(), start)
        self.assertLessEqual(brain.phantom_rate(), .30)

    def test_scenarios_add_unforeseen_rivals_without_shifting_other_draws(self):
        import random
        from game import Rules
        from sim import Market
        market = Market('phantom', rivals=3, days=4)
        brain = Brain(samples=300, small_crowds=True)
        state = market.state()
        brain.decide(state)
        state = market.play(brain.last_order)
        brain.observe(state)  # rival history is what unforeseen rivals are drawn from
        state['order_for_day'] = 3
        rules, counts = Rules.read(state), dict.fromkeys(Rules.read(state).spots, 100.)
        brain.phantom = {'events': 0., 'exposures': 1000.}  # rate floor
        floor = brain.scenarios(state, rules, counts, random.Random(1))
        brain.phantom = {'events': 300., 'exposures': 1000.}  # rate .30
        high = brain.scenarios(state, rules, counts, random.Random(1))
        self.assertTrue(any(r['id'] == 'phantom' for row in brain.scenario_rivals for r in row))
        total = lambda sc: sum(len(p) for placements, _, _, _ in sc for p in placements.values())
        self.assertGreater(total(high), total(floor))
        self.assertEqual([c for _, c, _, _ in floor], [c for _, c, _, _ in high])  # crowds are unchanged

    def test_checkpoint_roundtrip_and_old_checkpoints(self):
        brain = Brain(samples=4)
        brain.phantom = {'events': 2., 'exposures': 30.}
        other = Brain(samples=4); other.restore(brain.dump())
        self.assertEqual(other.phantom, brain.phantom)
        data = brain.dump(); data.pop('phantom')
        legacy = Brain(samples=4); legacy.restore(data)
        self.assertIn('events', legacy.phantom)


class LearnedSurpriseRateTests(unittest.TestCase):
    def test_rates_start_at_the_documented_values_and_follow_what_is_observed(self):
        brain = Brain(samples=4)
        self.assertAlmostEqual(brain.surprise_rate('fire', .06), .06)
        brain.surprises = {'days': 60, 'fire': 0, 'give': 30}
        self.assertLess(brain.surprise_rate('fire', .06), .06)
        self.assertGreater(brain.surprise_rate('give', .10), .10)
        other = Brain(samples=4); other.restore(brain.dump())
        self.assertEqual(other.surprises, brain.surprises)

    def test_moments_in_last_day_are_counted_once_per_day(self):
        from sim import Market
        market = Market('surprise-count', rivals=3, days=4, features={'fair_choice': True, 'walk_over': True, 'salvage': True})
        brain = Brain(samples=4, small_crowds=True)
        state = market.state()
        order = brain.decide(state)
        state = market.play(order)
        state['features'] = dict(state.get('features') or {}, surprises=True)
        state['last_day']['moments'] = [{'kind': 'firedrill', 'spot': 'thompson'}, {'kind': 'giveaway', 'spot': 'oval'}]
        brain.observe(state)
        brain.observe(state)  # the same day again must not be counted twice
        self.assertEqual(brain.surprises, {'days': 1, 'fire': 1, 'give': 1})

class DataDrivenCrewSearchTests(unittest.TestCase):
    def test_wage_range_follows_what_the_market_pays_not_a_fixed_ceiling(self):
        model = CrewModel(); model.observe(state(), 'me')
        _, wages = model.search_space(state(rivals=((11., 3), (10., 4))), 6., 2)
        self.assertGreaterEqual(max(wages), 12.)     # a dollar above the best wage anyone pays
        self.assertEqual(min(wages), 5.)
        _, rich = model.search_space(state(rivals=((20., 3),)), 6., 2)
        self.assertGreaterEqual(max(rich), 21.)      # no hand-set $12.50 ceiling
        _, quiet = model.search_space(state(rivals=()), 6., 2)
        self.assertLess(max(quiet), max(rich))

    def test_spare_headcount_is_chosen_only_when_learned_risk_pays_for_it(self):
        def market():
            s = state(rivals=((11., 3), (10., 4)))
            s['labor'] = {'unemployed': 0, 'lowest_asking_wage': 9., 'median_asking_wage': 10.}  # nobody to refill with
            return s
        calm, risky = CrewModel(), CrewModel()
        for model in (calm, risky):
            model.observe(market(), 'me')
        calm.poach = {'events': .1, 'exposures': 100.}
        risky.poach = {'events': 9., 'exposures': 10.}   # rivals really do poach us
        order = {'price': 10., 'meals': 90}
        low = calm.plan(market(), order, [88]*8, 'me')
        high = risky.plan(market(), order, [88]*8, 'me')
        # Learned poaching risk is answered with spare staff or, more cheaply, a wage that removes the exposure.
        self.assertTrue(high['wage'] > low['wage'] or high['headcount'] > low['headcount'])
        self.assertEqual(high['rivals_exposed'], 0) if high['wage'] > low['wage'] else None
        self.assertLessEqual(high['headcount'], 2 + 4)

    def test_rules_bound_the_search(self):
        s = state(); s['rules'] = {'staff': {'wage_min': 5, 'wage_max': 9, 'staff_max': 3}}
        model = CrewModel(); model.observe(s, 'me')
        heads, wages = model.search_space(s, 6., 2)
        self.assertLessEqual(max(heads), 3)
        self.assertLessEqual(max(wages), 9)


class WageFloorTests(unittest.TestCase):
    def test_wage_is_never_cut_below_the_games_default(self):
        s = state(); s['rules'] = {'staff': {'wage_min': 5, 'wage_max': 60, 'default_wage': 6, 'staff_max': 10}}
        model = CrewModel(); model.observe(s, 'me')
        _, wages = model.search_space(s, 6., 2)
        self.assertEqual(min(wages), 6.)
