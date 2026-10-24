import copy
import json
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
    def test_feature_toggle_replans_even_with_already_accepted_order(self):
        class Client:
            def __init__(self): self.posts = 0
            def call(self, method, path, body=None, **kwargs):
                if method == 'GET':
                    if self.posts >= 2: raise Stop()
                    st = state('live')
                    st['features'] = {'fair_choice': self.posts > 0}
                    return 200, st
                self.posts += 1
                return 200, {}
        client, brain = Client(), StubBrain()
        runner = Runner(client, brain)
        with self.assertRaises(Stop): runner.play()
        self.assertEqual(client.posts, 2)
        self.assertEqual(brain.calls, 2)
        self.assertTrue(runner.accepted_features['fair_choice'])

    def test_forbidden_order_is_not_retried_tonight(self):
        class Client:
            def __init__(self): self.posts = 0; self.gets = 0
            def call(self, method, path, body=None, **kwargs):
                if method == 'GET':
                    self.gets += 1
                    if self.gets > 3: raise Stop()
                    return 200, state()
                self.posts += 1
                return 403, {'error': 'your truck was wrecked in the riot'}
        client = Client()
        with self.assertRaises(Stop): Runner(client, StubBrain()).play()
        self.assertEqual(client.posts, 1)

    def test_idle_state_records_health_without_submitting(self):
        class Client:
            def __init__(self): self.calls = 0
            def call(self, method, path, body=None, **kwargs):
                self.calls += 1
                if self.calls > 1: raise Stop()
                assert method == 'GET'
                st = state('live'); st['phase'] = 'day'; st['orders_open'] = False
                return 200, st
        with tempfile.TemporaryDirectory() as folder:
            runner = Runner(Client(), StubBrain(), Path(folder)/'runtime-v1.json')
            with self.assertRaises(Stop): runner.play()
            health = json.loads((Path(folder)/'connection-status.json').read_text())
            self.assertEqual(health['phase'], 'day')
            self.assertIsNone(health['accepted'])

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


class ObserveGuardTests(unittest.TestCase):
    def test_a_failing_day_observation_does_not_end_the_run(self):
        class Client:
            def __init__(self): self.gets = 0
            def call(self, method, path, body=None, **kwargs):
                self.gets += 1
                if self.gets > 3: raise Stop()
                st = state('live'); st['phase'] = 'day'; st['orders_open'] = False
                st['last_day'] = {'day': 5, 'spots': {}}
                return 200, st
        class Brain(StubBrain):
            def observe(self, st): raise TypeError('boom')
        client = Client()
        with self.assertRaises(Stop): Runner(client, Brain()).play()
        self.assertEqual(client.gets, 4)
