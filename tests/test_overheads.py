import unittest
from bot import Runner, fallback
from game import Rules, SPOTS
from overheads import Overheads
from politics import choose_vote, law_terms
from sim import Market
from strategy import Brain


def with_rules(**flags):
    state = Market('overheads', rivals=3, days=3).state()
    state['features'] = dict(state.get('features') or {}, **flags)
    state['order_for_day'] = 2
    return state


class OverheadTests(unittest.TestCase):
    def test_nothing_changes_while_rules_are_off(self):
        state = with_rules()
        state['you']['staff'] = {'employees': 4, 'wage': 6, 'capacity': 200}
        oh = Overheads(state, 'oval', False)
        self.assertEqual(oh.fixed, 0)
        self.assertIsNone(oh.capacity)
        self.assertEqual(oh.profit(60, 60, 8.5, 'premium', 5.5), 180)

    def test_wages_moving_and_capacity(self):
        state = with_rules(staff=True, moving_costs=True)
        state['you'].update(last_spot='oval', staff={'employees': 2, 'target': 2, 'wage': 6, 'capacity': 100})
        state['rules']['move_costs'] = {'oval': {'stadium': 8, 'oval': 0}}
        stay, move = Overheads(state, 'oval', False), Overheads(state, 'stadium', False)
        self.assertEqual((stay.fixed, move.fixed), (12., 20.))
        self.assertEqual(stay.capacity, 100)
        self.assertEqual(stay.profit(60, 60, 8.5, 'premium', 5.5), 168)
        self.assertEqual(stay.profit(0, 0, 8.5, 'premium', 5.5), 0)  # no order, no wages
        order = Brain(samples=4, small_crowds=True).decide(state)
        self.assertLessEqual(order['meals'], 100)

    def test_first_day_out_is_free(self):
        state = with_rules(moving_costs=True)
        state['you']['last_spot'] = None
        state['rules']['move_costs'] = {'oval': {'stadium': 8}}
        self.assertEqual(Overheads(state, 'stadium', False).fixed, 0)

    def test_laws_change_price_and_profit(self):
        floor = law_terms([{'key': 'price_floor', 'text': 'Price floor: ...', 'params': {'min_price': 7.5}}])
        self.assertEqual(floor['floor'], 7.5)
        state = with_rules(elections=True)
        state['politics'] = {'laws': [{'key': 'junk_tax', 'text': 'Junk food tax: ...'}]}
        oh = Overheads(state, 'oval', False)
        self.assertEqual(oh.profit(60, 60, 8.5, 'premium', 5.5), 180 + 30)
        self.assertEqual(oh.profit(60, 60, 4.0, 'cheap', 2.5), 60*4 - 150 - 30)

    def test_vote_only_when_clearly_helpful(self):
        order = {'spot': 'oval', 'quality': 'premium', 'price': 8.5, 'meals': 80}
        state = {'politics': {'election': {'id': 'e1', 'candidates': [
            {'id': 'c1', 'policies': [{'key': 'luxury_tax', 'text': 'Luxury tax: ...'}]},
            {'id': 'c2', 'policies': [{'key': 'junk_tax', 'text': 'Junk food tax: ...'}]},
            {'id': 'c3', 'policies': [{'key': 'price_cap', 'text': 'Price cap: ...'}]}]}}}
        self.assertEqual(choose_vote(state, order), 'c2')
        self.assertIsNone(choose_vote(state, dict(order, meals=10)))
        self.assertIsNone(choose_vote(state, None))

    def test_unknown_spot_does_not_break_planning(self):
        state = with_rules()
        state['rules']['spots'] = list(state['rules']['spots']) + ['fisher']
        state['forecast']['expected_customers']['fisher'] = 5
        order = Brain(samples=4, small_crowds=True).decide(state)
        self.assertIn(order['spot'], SPOTS)
        self.assertIn(fallback(state)['spot'], SPOTS)


class FootballTests(unittest.TestCase):
    def test_game_day_opponent_and_championship_events_are_recognized(self):
        from game import events
        self.assertEqual(events([{'key': 'gameday', 'name': 'Game day vs Iowa', 'spot': 'stadium'}]), ['game'])
        self.assertEqual(events([{'key': 'championship', 'name': 'National Championship'}]), ['game'])
        self.assertEqual(events([]), [])


class VoteRunnerTests(unittest.TestCase):
    def test_votes_once_per_election_and_never_raises(self):
        class Client:
            def __init__(self): self.posts = []
            def call(self, method, path, body=None, timeout=30):
                self.posts.append((path, body)); return 200, {}
        class StubBrain:
            last_order = {'spot': 'oval', 'quality': 'premium', 'price': 8.5, 'meals': 80}
        client = Client()
        runner = Runner(client, StubBrain())
        state = {'mode': 'live', 'market': 1, 'you': {}, 'politics': {'election': {'id': 'e1', 'candidates': [
            {'id': 'c2', 'policies': [{'key': 'junk_tax', 'text': 'Junk food tax: ...'}]}]}}}
        runner.vote(state, '/api/v1')
        runner.vote(state, '/api/v1')
        self.assertEqual(client.posts, [('/api/v1/vote', {'candidate': 'c2'})])
        runner.vote({'mode': 'live', 'market': 1, 'politics': {'election': {'id': 'e2', 'candidates': 'bad'}}}, '/api/v1')


if __name__ == '__main__':
    unittest.main()
