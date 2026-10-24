import copy
import tempfile
import unittest
from pathlib import Path
from bot import Runner, make_brain
from game import SPOTS


def state(mode='practice', market=1, day=1):
    return {'version': market*10+day, 'mode': mode, 'market': market, 'phase': 'night',
            'orders_open': True, 'order_for_day': day, 'days': 2,
            'forecast': {'expected_customers': dict.fromkeys(SPOTS, 30)},
            'you': {'id': 'us', 'cash': 0, 'reputation': dict.fromkeys(SPOTS, 50)},
            'trucks': [{'id': 'us'}]}


class Stop(Exception): pass


class StubBrain:
    diagnostics = {}
    def __init__(self): self.calls = 0
    def decide(self, st):
        self.calls += 1
        return {'spot': 'oval', 'quality': 'standard', 'price': 7, 'meals': 10}
    def observe(self, st): pass


class RunnerTests(unittest.TestCase):
    def test_timeout_retries_cached_order_with_immediate_state_poll(self):
        class Client:
            def __init__(self): self.gets = []; self.posts = []
            def call(self, method, path, body=None, **kwargs):
                if method == 'GET':
                    self.gets.append(path)
                    return 200, state()
                self.posts.append(body)
                if len(self.posts) == 1: raise TimeoutError()
                final = state(); final['phase'] = 'over'
                return 200, {'state': final}
        client, brain = Client(), StubBrain()
        Runner(client, brain).play('/api/v1/practice/test')
        self.assertEqual(brain.calls, 1)
        self.assertEqual(client.posts[0], client.posts[1])
        self.assertTrue(all(x.endswith('since=0') for x in client.gets))

    def test_final_market_resets_do_not_skip_day_one(self):
        class Client:
            def __init__(self): self.next = state('final', 1); self.posts = []
            def call(self, method, path, body=None, **kwargs):
                if method == 'GET':
                    if self.next is None: raise Stop()
                    return 200, self.next
                self.posts.append(body)
                self.next = state('final', 2) if len(self.posts) == 1 else None
                return 200, {}
        client, brain = Client(), StubBrain()
        with self.assertRaises(Stop): Runner(client, brain).play()
        self.assertEqual(len(client.posts), 2)

    def test_conflict_refreshes_state_instead_of_marking_accepted(self):
        class Client:
            def __init__(self): self.gets = []; self.posts = 0
            def call(self, method, path, body=None, **kwargs):
                if method == 'GET':
                    self.gets.append(path); return 200, state()
                self.posts += 1
                if self.posts == 1: return 409, {}
                final = state(); final['phase'] = 'over'
                return 200, {'state': final}
        client = Client()
        Runner(client, StubBrain()).play('/api/v1/practice/test')
        self.assertEqual(client.posts, 2)
        self.assertTrue(all(x.endswith('since=0') for x in client.gets))


if __name__ == '__main__': unittest.main()
