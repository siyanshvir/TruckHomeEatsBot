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
