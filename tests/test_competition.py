import json
import unittest
from competition import Competition, LIMIT


def day(layout, me='me', profit=None):
    spots = {s: {'customers': 100, 'hungry': 50, 'trucks': [
        {'id': i, 'truck_name': i, 'price': p, 'quality': 3, 'meals': 40, 'sold': 30} for i, p in ts]}
        for s, ts in layout.items()}
    you = {'id': me}
    if profit is not None:
        you['profit'] = profit
    return {'spots': spots, 'you': you}


ORDER = {'spot': 'oval', 'quality': 3, 'price': 10., 'meals': 40}
STATE = {'day': 10, 'days': 30, 'you': {'reputation': {'oval': 60., 'union': 60.}}}


class CompetitionTests(unittest.TestCase):
    def test_no_data_zero_apart_from_small_decaying_bonus(self):
        c = Competition()
        self.assertEqual(c.penalty('oval', STATE), 0.)
        early = c.penalty('oval', {'day': 1, 'days': 30})
        late = c.penalty('oval', {'day': 29, 'days': 30})
        self.assertLess(early, 0); self.assertGreaterEqual(early, -15)
        self.assertGreater(late, early)
        c.observe(day({'oval': [('me', 10.)]}), ORDER, {'profit': 100})
        self.assertGreater(c.penalty('oval', {'day': 1, 'days': 30}), early)  # decays with visits

    def test_repeated_shortfalls_raise_penalty(self):
        c = Competition(); prev = c.penalty('oval', STATE)
        for _ in range(5):
            c.observe(day({'oval': [('me', 10.)]}, profit=90.), ORDER, {'profit': 120.})
            now = c.penalty('oval', STATE)
            self.assertGreater(now, prev); prev = now
        self.assertGreater(prev, 20)
        self.assertEqual(c.penalty('union', STATE), 0.)

    def test_overperformance_is_a_small_bonus(self):
        c = Competition()
        for _ in range(5):
            c.observe(day({'oval': [('me', 10.)]}, profit=200.), ORDER, {'profit': 100.})
        self.assertLess(c.penalty('oval', STATE), 0)
        self.assertGreaterEqual(c.penalty('oval', STATE), -15)

    def test_follower_raises_pressure_at_spot_we_left(self):
        c = Competition()
        base = None
        for i in range(4):
            if i % 2 == 0:   # we sit at oval, rival at union
                c.observe(day({'oval': [('me', 10.)], 'union': [('r', 9.)]}, profit=100.), ORDER, {'profit': 100})
            else:            # we move to union, rival moves to oval (the spot we left)
                c.observe(day({'oval': [('r', 9.)], 'union': [('me', 10.)]}, profit=100.),
                          dict(ORDER, spot='union'), {'profit': 100})
            if base is None:
                base = c.penalty('oval', STATE)
        self.assertGreater(c.penalty('oval', STATE), base + 3)
        self.assertGreater(c.spots['oval']['follow'], 0)

    def test_bounded(self):
        c = Competition()
        for _ in range(20):
            c.observe(day({'oval': [('me', 10.)] + [('r%d' % i, 1.) for i in range(8)]}, profit=-1e9),
                      ORDER, {'profit': 1e9})
        for b in (0, .5, 1, 5, 'x', None):
            self.assertLessEqual(abs(c.scaled('oval', STATE, b)), LIMIT)
        self.assertLessEqual(abs(c.penalty('oval', STATE)), LIMIT)
        self.assertEqual(c.scaled('oval', STATE, 0.), 0.)

    def test_roundtrip(self):
        c = Competition()
        for i in range(3):
            c.observe(day({'oval': [('me', 10.)], 'union': [('r', 9.)]}, profit=10.), ORDER, {'profit': 90})
        r = Competition(json.loads(json.dumps(c.to_dict())))
        self.assertEqual(r.to_dict(), c.to_dict())
        self.assertEqual(r.penalty('oval', STATE), c.penalty('oval', STATE))

    def test_malformed_ignored(self):
        c = Competition()
        for args in ((None, None, None), ('x', 1, 2), ({}, ORDER, {}), ({'spots': 3}, ORDER, {'profit': 'a'}),
                     ({'spots': {'oval': {'trucks': 'z'}}}, ORDER, {'profit': float('nan')}),
                     ({'spots': {'oval': {'trucks': [1, None]}}, 'you': 5}, ORDER, [])):
            c.observe(*args)
        for state in (None, 'x', {'day': 'a', 'you': {'reputation': 4}}, {'days': 0}):
            self.assertIsInstance(c.penalty('oval', state), float)
        self.assertIsInstance(Competition({'spots': 4, 'prev_loc': [1]}).penalty('a', None), float)
        self.assertIsInstance(Competition({'spots': {'a': {'gap': 'x', 'n': None}}}).penalty('a', None), float)

    def test_new_market_resets_visits_keeps_learning(self):
        c = Competition()
        for _ in range(4):
            c.observe(day({'oval': [('me', 10.)]}, profit=0.), ORDER, {'profit': 100})
        before = c.penalty('oval', {'day': 1, 'days': 30})
        c.new_market()
        self.assertEqual(c.spots['oval']['visits'], 0)
        self.assertGreater(c.penalty('oval', STATE), 0)
        self.assertLess(c.penalty('oval', {'day': 1, 'days': 30}), before)


if __name__ == '__main__':
    unittest.main()
