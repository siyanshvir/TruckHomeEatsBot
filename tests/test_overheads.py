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

    def test_vote_follows_win_odds_and_protest_cost(self):
        from politics import vote_analysis
        order = {'spot': 'thompson', 'quality': 'premium', 'price': 10.25, 'meals': 90}
        flat = {'key': 'flat_fee', 'text': 'Flat fee: ...', 'params': {'revenue_share': .1, 'above': 500, 'flat': -20}}
        curfew = {'key': 'curfew', 'text': 'North campus curfew: ...'}
        quiet = {'key': 'quiet_core', 'text': 'Quiet core: ...'}
        state = {'days': 65, 'you': {'id': 'me', 'cash': 100}, 'trucks': [{'id': t, 'cash': 500} for t in ('me', 'a', 'b', 'c')],
                 'politics': {'election': {'id': 'e1', 'vote_day': 27, 'tally': {'c1': 1, 'c3': 1},
                              'votes': {'a': 'c1', 'b': 'c3'}, 'candidates': [
            {'id': 'c1', 'policies': [flat, curfew]}, {'id': 'c3', 'policies': [flat, quiet]}]}}}
        rows = {r['id']: r for r in vote_analysis(state, order)['rows']}
        self.assertGreater(rows['c1']['p_win_vote'], rows['c1']['p_win_abstain'])
        self.assertGreater(rows['c1']['daily_value'], 0)
        self.assertLess(rows['c3']['daily_value'], 0)
        self.assertEqual(choose_vote(state, order), 'c1')
        self.assertIsNone(choose_vote(state, dict(order, meals=10)))
        self.assertIsNone(choose_vote(state, None))
        self.assertIsNone(choose_vote({'politics': {}}, order))

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
            last_order = {'spot': 'thompson', 'quality': 'premium', 'price': 10.25, 'meals': 90}
        client = Client()
        runner = Runner(client, StubBrain())
        flat = {'key': 'flat_fee', 'text': 'Flat fee: ...', 'params': {'revenue_share': .1, 'above': 500, 'flat': -20}}
        state = {'mode': 'live', 'market': 1, 'days': 65, 'you': {'id': 'me', 'cash': 100},
                 'trucks': [{'id': t, 'cash': 500} for t in ('me', 'a', 'b', 'c')],
                 'politics': {'election': {'id': 'e1', 'vote_day': 27, 'tally': {'c2': 1, 'c3': 1},
                              'votes': {'a': 'c2', 'b': 'c3'}, 'candidates': [
            {'id': 'c2', 'policies': [flat, {'key': 'curfew', 'text': 'North campus curfew: ...'}]},
            {'id': 'c3', 'policies': [flat, {'key': 'quiet_core', 'text': 'Quiet core: ...'}]}]}}}
        runner.vote(state, '/api/v1')
        runner.vote(state, '/api/v1')
        self.assertEqual(client.posts, [('/api/v1/vote', {'candidate': 'c2'})])
        runner.vote({'mode': 'live', 'market': 1, 'politics': {'election': {'id': 'e2', 'candidates': 'bad'}}}, '/api/v1')


if __name__ == '__main__':
    unittest.main()


class GameDayLawTests(unittest.TestCase):
    def test_game_day_laws_are_read_from_params_and_weighted_by_game_day_rate(self):
        from politics import vote_analysis
        order = {'spot': 'thompson', 'quality': 'premium', 'price': 10., 'meals': 90}
        crackdown = {'key': 'crackdown', 'params': {'game_crowd': {'stadium': .85, 'highst': .85, 'schott': .85, 'other': 1.1}, 'riot': 0}}
        tailgate = {'key': 'tailgate', 'params': {'game_crowd': {'stadium': 1.3, 'highst': 1.3, 'schott': 1.3, 'other': .9}, 'riot': 1.5}}
        state = {'days': 65, 'you': {'id': 'me', 'cash': 100}, 'trucks': [{'id': t, 'cash': 500} for t in ('me', 'a', 'b', 'c', 'd', 'e')],
                 'politics': {'election': {'id': 'e2', 'vote_day': 34, 'tally': {}, 'votes': {},
                              'candidates': [{'id': 'c1', 'policies': [crackdown]}, {'id': 'c2', 'policies': [tailgate]}]}}}
        low = {r['id']: r for r in vote_analysis(state, order, .1)['rows']}
        high = {r['id']: r for r in vote_analysis(state, order, .5)['rows']}
        self.assertGreater(low['c1']['daily_value'], 0)   # Thompson is "other": crackdown helps
        self.assertLess(low['c2']['daily_value'], 0)      # tailgate hurts
        self.assertGreater(high['c1']['daily_value'], low['c1']['daily_value'])
        self.assertLess(high['c2']['daily_value'], low['c2']['daily_value'])
        from politics import law_terms
        self.assertEqual(law_terms([crackdown])['riot'], 0)

    def test_any_law_with_crowd_params_is_used_even_with_an_unknown_name(self):
        from politics import law_terms
        terms = law_terms([{'key': 'mystery_rule', 'params': {'crowd': {'thompson': 1.2}}}])
        self.assertEqual(terms['crowd'], {'thompson': 1.2})

    def test_brain_estimates_game_day_rate_from_history(self):
        brain = Brain(samples=4)
        self.assertAlmostEqual(brain.game_day_rate(), .25)
        brain.history = {'x': [{'day': d, 'context': {'events': ['game'] if d % 3 == 0 else []}} for d in range(1, 31)]}
        self.assertGreater(brain.game_day_rate(), .25)
