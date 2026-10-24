import unittest
from calibration import Calibrator


def check(price=9.5, ceiling=11., expected=100., actual=50., meals=120, quality='standard', **kw):
    d = {'spot': 'stadium', 'quality': quality, 'price': price, 'value_ceiling': ceiling,
         'expected_demand': expected, 'actual_sales': actual, 'meals': meals, 'sellout': False}
    d.update(kw)
    return d


class CalibrationTests(unittest.TestCase):
    def test_neutral_with_no_data(self):
        c = Calibrator()
        self.assertEqual(c.factor('stadium', 'standard', 10., 11.), 1.)
        self.assertEqual(c.factor('stadium', 'bogus', 10., 11.), 1.)
        self.assertEqual(c.factor('stadium', 'standard', 10., 0), 1.)

    def test_high_price_misses_lower_only_similar_ratio(self):
        c = Calibrator()
        for _ in range(3):
            c.observe(check(price=9.5, ceiling=11.))  # r=.86, half of demand
        self.assertLess(c.factor('stadium', 'standard', 9.6, 11.), .8)
        self.assertGreater(c.factor('stadium', 'standard', 5., 11.), .97)
        self.assertGreaterEqual(c.factor('stadium', 'standard', 5., 11.), c.factor('stadium', 'standard', 8., 11.))

    def test_sellouts_never_lower(self):
        c = Calibrator()
        for _ in range(5):
            c.observe(check(actual=60, meals=60))
            c.observe(check(actual=30, meals=80, sellout=True))
        self.assertEqual(c.factor('s', 'standard', 9.5, 11.), 1.)
        self.assertEqual(c.report()['observations'], 0)

    def test_sellout_can_raise(self):
        c = Calibrator()
        for _ in range(3):
            c.observe(check(actual=60))
        low = c.factor('s', 'standard', 9.5, 11.)
        for _ in range(3):
            c.observe(check(actual=115, meals=115))
        self.assertGreater(c.factor('s', 'standard', 9.5, 11.), low)

    def test_roundtrip(self):
        c = Calibrator()
        for a in (40, 55, 70):
            c.observe(check(actual=a))
            c.observe(check(price=6., actual=95, quality='cheap'))
        d = Calibrator(c.to_dict())
        self.assertEqual(d.to_dict(), c.to_dict())
        for q, p in (('standard', 9.5), ('cheap', 6.), ('premium', 14.)):
            self.assertEqual(c.factor('x', q, p, 11.), d.factor('x', q, p, 11.))
        self.assertEqual(c.report(), d.report())

    def test_malformed_ignored(self):
        c = Calibrator()
        for bad in (None, 5, {}, check(actual='x'), check(price=-1), check(expected=0),
                    check(quality='nope'), check(actual=float('nan')), check(ceiling=0), {'quality': 'standard'}):
            c.observe(bad)
        self.assertEqual(c.report()['observations'], 0)
        for bad in (None, 5, {'cells': 7}, {'cells': {'bad': [1, 2]}}, {'cells': {'standard|1': ['a', 1]}}):
            self.assertEqual(Calibrator(bad).factor('s', 'standard', 9., 11.), 1.)
        self.assertEqual(c.factor('s', 'standard', None, 11.), 1.)

    def test_forgetting_allows_recovery(self):
        c = Calibrator()
        for _ in range(4):
            c.observe(check(actual=45))
        low = c.factor('s', 'standard', 9.5, 11.)
        for _ in range(8):
            c.observe(check(actual=100))
        self.assertGreater(c.factor('s', 'standard', 9.5, 11.), low + .2)

    def test_day28_replay(self):
        c = Calibrator()
        for _ in range(3):
            c.observe(check(price=10.92, ceiling=12.85, expected=110, actual=53, meals=90))
        f = c.factor('thompson', 'standard', 10.92, 12.85)
        self.assertTrue(.5 <= f <= .8, f)
        for _ in range(20):
            c.observe(check(price=10.92, ceiling=12.85, expected=110, actual=53, meals=90))
        self.assertGreaterEqual(c.factor('thompson', 'standard', 10.92, 12.85), .3)
        self.assertLessEqual(c.factor('thompson', 'standard', 10.92, 12.85), .6)

    def test_output_clamped(self):
        c = Calibrator()
        for _ in range(50):
            c.observe(check(actual=0))
        self.assertGreaterEqual(c.factor('s', 'standard', 9.5, 11.), .3)
        c = Calibrator()
        for _ in range(50):
            c.observe(check(actual=500, meals=500))
        self.assertLessEqual(c.factor('s', 'standard', 9.5, 11.), 1.15)


if __name__ == '__main__':
    unittest.main()
