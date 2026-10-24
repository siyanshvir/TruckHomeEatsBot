import json
import os
import tempfile
import unittest
from collections import Counter
from controller import ARMS, MetaController, Knobs, load_priors


def valid(k):
    return all(getattr(k, n) in ARMS[n] for n in ARMS)


class ControllerTests(unittest.TestCase):
    def test_no_data_valid_and_mostly_middle(self):
        c = MetaController(seed=1)
        picks = [c.choose() for _ in range(300)]
        self.assertTrue(all(valid(k) for k in picks))
        for i, n in enumerate(ARMS):
            share = Counter(getattr(k, n) for k in picks)[ARMS[n][1]] / 300.
            self.assertGreater(share, .55)
            self.assertLess(share, 1.0)  # still explores

    def test_greedy_deterministic_and_middle(self):
        c = MetaController(seed=3)
        self.assertEqual(c.choose(greedy=True), Knobs(1.0, 0.5, 0.10))
        self.assertEqual(c.choose(greedy=True), c.choose(greedy=True))

    def test_seeded_deterministic(self):
        a, b = MetaController(seed=7), MetaController(seed=7)
        self.assertEqual([a.choose() for _ in range(20)], [b.choose() for _ in range(20)])

    def test_positive_reward_shifts_choices(self):
        c = MetaController(seed=2)
        target = Knobs(1.10, 0.75, 0.03)
        for _ in range(40):
            c.update(target, 30.); c.update(Knobs(1.0, 0.5, 0.10), -10.)
        picks = [c.choose() for _ in range(200)]
        self.assertGreater(sum(k.price_aggr == 1.10 for k in picks), 120)
        self.assertEqual(c.choose(greedy=True), target)

    def test_discounting_readapts(self):
        c = MetaController(seed=4)
        a, b = Knobs(1.10, 0.75, 0.30), Knobs(0.90, 0.0, 0.03)
        for _ in range(60):
            c.update(a, 25.)
        self.assertEqual(c.choose(greedy=True), a)
        for _ in range(120):
            c.update(b, 25.); c.update(a, -25.)
        self.assertEqual(c.choose(greedy=True), b)

    def test_never_outside_arms_even_with_garbage(self):
        c = MetaController(seed=5)
        for bad in (None, 'x', float('nan'), float('inf')):
            c.update(Knobs(1.0, .5, .1), bad)
        c.update(None, 3); c.update((1, 2), 3); c.update(Knobs(5, 5, 5), 1e9)
        for _ in range(100):
            self.assertTrue(valid(c.choose({'losing_streak': 9})))

    def test_roundtrip(self):
        c = MetaController(seed=6)
        for i in range(15):
            c.update(c.choose(), i - 7.)
        d = json.loads(json.dumps(c.to_dict()))
        r = MetaController(d)
        self.assertEqual(r.to_dict(), c.to_dict())
        self.assertEqual([r.choose() for _ in range(10)], [c.choose() for _ in range(10)])

    def test_malformed_data_ignored(self):
        for bad in ({'arms': 3}, {'arms': {'risk': [1]}}, {'arms': {'risk': [{'n': 'x'}] * 3}}, 'junk', []):
            self.assertTrue(valid(MetaController(bad).choose()))

    def test_new_market_keeps_means_but_loosens(self):
        c = MetaController(seed=8)
        t = Knobs(1.10, 0.75, 0.30)
        for _ in range(30):
            c.update(t, 20.)
        before = c.means()['risk'][2]; var = c.posterior(c.arms['risk'][2])[1]
        c.new_market()
        self.assertGreater(c.posterior(c.arms['risk'][2])[1], var)
        self.assertAlmostEqual(c.means()['risk'][2], before, delta=.15)

    def test_load_priors(self):
        self.assertEqual(load_priors('/nonexistent/final_priors.json'), {})
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'p.json')
            with open(p, 'w') as f:
                json.dump({'controller': {'risk': {'0.3': {'mean': 2.0, 'var': .05, 'n': 9}}}}, f)
            pri = load_priors(p)
            self.assertEqual(MetaController(priors=pri).choose(greedy=True).risk, 0.30)
            with open(p, 'w') as f:
                f.write('{bad')
            self.assertEqual(load_priors(p), {})
            with open(p, 'w') as f:
                json.dump({'x': 1}, f)
            self.assertEqual(load_priors(p), {})


if __name__ == '__main__':
    unittest.main()
