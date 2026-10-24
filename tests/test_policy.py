import json
import random
import unittest
from pathlib import Path
from bot import make_brain
from game import SPOTS, QUALITIES, Rules
from learning import Learner
from strategy import Brain
from sim import Market


class PolicyTests(unittest.TestCase):
    def state(self):
        return Market('unit', rivals=5, days=3).state()

    def test_every_location_and_quality_evaluated_and_legal(self):
        brain = Brain(samples=4)
        state = self.state()
        Rules.read(state).validate(brain.decide(state))
        self.assertEqual(brain.diagnostics['coverage'], {s: list(QUALITIES) for s in SPOTS})
        self.assertEqual(len(brain.diagnostics['ranking']), 6)

    def test_zero_forecast_is_preserved_and_zero_food_is_legal(self):
        state = self.state()
        state['forecast']['expected_customers'] = dict.fromkeys(SPOTS, 0)
        brain = Brain(samples=4)
        self.assertEqual(brain.decide(state)['meals'], 0)
        self.assertEqual(brain.forecast(state, Rules.read(state)), dict.fromkeys(SPOTS, 0))

    def test_quiet_uncontested_location_beats_busy_blocked_location(self):
        state = self.state()
        state['forecast']['expected_customers'] = dict.fromkeys(SPOTS, 0)
        state['forecast']['expected_customers'].update(highst=200, oval=100)
        state['order_for_day'] = state['days']  # No future reputation incentive.
        brain = Brain(samples=4)
        brain.observe(state)
        # Known contest at highst: cheap premium leaves almost no first-choice demand.
        scenarios = []
        for _ in range(4):
            placements = {s: [] for s in SPOTS}
            placements['highst'] = [(5.6, 'premium', 100, 500)]
            crowds = dict(state['forecast']['expected_customers'])
            models = {s: (12, 3, (.4, .4, .2)) for s in SPOTS}
            scenarios.append((placements, crowds, models, 5.))
        brain.scenarios = lambda *args: scenarios
        self.assertEqual(brain.decide(state)['spot'], 'oval')

    def test_all_qualities_can_win(self):
        for desired in QUALITIES:
            state = self.state()
            state['order_for_day'] = state['days']
            # Make one quality cost-effective; prevent an implementation premium lock.
            state['rules']['costs'] = {q: 1000. for q in QUALITIES}
            state['rules']['costs'][desired] = 2.
            state['trucks'] = [{'id': 'us'}]
            state['you']['reputation'] = dict.fromkeys(SPOTS, 100.)
            state['forecast']['expected_customers'] = dict.fromkeys(SPOTS, 150.)
            self.assertEqual(Brain(samples=4).decide(state)['quality'], desired)

    def test_learning_is_idempotent_and_market_reputation_resets(self):
        market = Market('reset', rivals=3, days=3)
        brain = Brain(samples=4)
        order = brain.decide(market.state())
        state = market.play(order)
        brain.observe(state)
        count = brain.learner.rows
        brain.observe(state)
        self.assertEqual(brain.learner.rows, count)
        self.assertTrue(brain.history)
        state['market'] = 'new'
        state['last_day'] = None
        brain.observe(state)
        self.assertEqual(brain.reputations, {})
        self.assertEqual(brain.learner.rows, count)

    def test_checkpoint_round_trip_and_reproducibility(self):
        market = Market('persist', rivals=3, days=3)
        brain = Brain(samples=4)
        market.play(brain.decide(market.state()))
        state = market.state()
        first = brain.decide(state)
        restored = Brain(samples=4)
        restored.restore(json.loads(json.dumps(brain.dump())))
        self.assertEqual(first, restored.decide(state))

    def test_reported_day_eight_loss_is_not_repeated(self):
        state = json.loads((Path(__file__).parent / 'fixtures' / 'day9.json').read_text())
        brain = make_brain()
        order = brain.decide(state)
        self.assertNotEqual(order['spot'], 'thompson')
        self.assertLess(order['price'], 12.)
        self.assertGreater(brain.diagnostics['chosen']['profit'], 0.)

    def test_sellout_learning_does_not_penalize_excess_demand(self):
        learner = Learner()
        learner.learn({'day': 1, 'spots': {'oval': {'customers': 1000,
                      'trucks': [{'id': 'x', 'quality': 'premium', 'price': 6, 'meals': 1, 'sold': 1}]}}})
        self.assertAlmostEqual(max(learner.weights['oval'])-min(learner.weights['oval']), 0.)


if __name__ == '__main__': unittest.main()
