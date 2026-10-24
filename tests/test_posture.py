import unittest
import posture
from posture import read, note_cash, NEUTRAL
from sim import Market
from strategy import Brain


def history(me_rate, rival_rates, days=8, me='me'):
    cash = {}
    for d in range(1, days + 1):
        cash[d] = {me: me_rate * d}
        for i, r in enumerate(rival_rates):
            cash[d]['r%d' % i] = r * d
    return cash


class PostureTests(unittest.TestCase):
    def test_neutral_without_enough_history(self):
        self.assertEqual(read({'day': 2}, history(100, [300], days=2), 'me'), NEUTRAL)
        self.assertEqual(read({'day': 8}, {}, 'me'), NEUTRAL)

    def test_behind_raises_risk_taking_and_roaming_ahead_lowers_them(self):
        behind = read({'day': 8}, history(100, [300, 250]), 'me')
        ahead = read({'day': 8}, history(400, [300, 250]), 'me')
        even = read({'day': 8}, history(300, [300, 250]), 'me')
        self.assertGreater(behind['pressure'], .5)
        self.assertLess(behind['risk_scale'], 1.)
        self.assertGreater(behind['upside'], 0)
        self.assertGreater(behind['explore'], 0)
        self.assertGreater(ahead['lead'], 0)
        self.assertGreater(ahead['risk_scale'], 1.)
        self.assertEqual(ahead['explore'], 0)
        self.assertLess(even['pressure'] + even['lead'], .1)
        self.assertTrue(.4 <= behind['risk_scale'] <= 1.5 and .4 <= ahead['risk_scale'] <= 1.5)

    def test_effects_are_bounded_and_scale_in_with_evidence(self):
        short = read({'day': 4}, history(0, [500], days=4), 'me')
        full = read({'day': 8}, history(0, [500], days=8), 'me')
        self.assertLess(short['pressure'], full['pressure'])
        self.assertLessEqual(full['upside'], posture.MAX_UPSIDE)
        self.assertLessEqual(full['explore'], posture.MAX_EXPLORE)

    def test_cash_history_is_recorded_and_trimmed(self):
        hist = {}
        for day in range(1, 20):
            note_cash(hist, {'day': day, 'trucks': [{'id': 'a', 'cash': day}], 'you': {'id': 'me', 'cash': 2 * day}})
        self.assertLessEqual(len(hist), posture.WINDOW + 3)
        self.assertEqual(hist[19], {'a': 19.0, 'me': 38.0})
        note_cash(hist, {'day': 'x'}); note_cash(hist, {})   # malformed input never raises

    def test_kill_switch(self):
        import os, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'profile.json'
            self.assertTrue(posture.enabled(path))
            path.write_text('{"posture": "off"}')
            self.assertFalse(posture.enabled(path))
        os.environ['NIGHTSHIFT_POSTURE'] = 'off'
        try:
            self.assertFalse(posture.enabled())
        finally:
            del os.environ['NIGHTSHIFT_POSTURE']

    def test_planner_is_unchanged_when_neutral_and_checkpoints_keep_history(self):
        market = Market('posture-plan', rivals=3, days=6, features={'fair_choice': True, 'walk_over': True, 'salvage': True})
        plain, shaded = Brain(samples=8, small_crowds=True), Brain(samples=8, small_crowds=True)
        state = market.state()
        self.assertEqual(plain.decide(state), shaded.decide(state))
        shaded.cash_history = history(100, [300, 250])
        other = Brain(samples=8, small_crowds=True)
        other.restore(shaded.dump())
        self.assertEqual(other.cash_history, shaded.cash_history)

    def test_being_behind_makes_the_planner_prefer_upside(self):
        market = Market('posture-risk', rivals=3, days=12, features={'fair_choice': True, 'walk_over': True, 'salvage': True})
        brain = Brain(samples=10, small_crowds=False)
        state = market.state()
        for _ in range(7):
            state = market.play(brain.decide(state))
        calm = brain.decide(dict(state))
        self.assertEqual(brain.posture['pressure'], 0.)
        brain.cash_history = history(0, [400, 300], days=7, me=state['you']['id'])
        brain.decide(dict(state))
        self.assertGreater(brain.diagnostics['posture']['pressure'], 0)


if __name__ == '__main__':
    unittest.main()


class CatchUpTests(unittest.TestCase):
    def test_a_big_cash_gap_with_few_days_left_raises_pressure_in_the_live_game_only(self):
        hist = history(300, [300, 280], days=8)
        for d in hist:                       # a rival is far ahead in cash but earns the same pace
            hist[d]['r0'] += 4000
        live = {'day': 8, 'days': 36, 'mode': 'live'}
        final = {'day': 8, 'days': 36, 'mode': 'final'}
        self.assertGreater(read(live, hist, 'me')['pressure'], read(final, hist, 'me')['pressure'])
        self.assertLessEqual(read(live, hist, 'me')['risk_scale'], 1.)
        self.assertEqual(read({'day': 8, 'days': 8, 'mode': 'live'}, hist, 'me')['pressure'],
                         read(final, hist, 'me')['pressure'])   # no days left: nothing to catch up


class FinalDefaultTests(unittest.TestCase):
    def test_final_markets_stay_neutral_unless_switched_on(self):
        import os
        hist = history(0, [400, 300], days=8)
        self.assertGreater(read({'day': 8, 'days': 30, 'mode': 'live'}, hist, 'me')['pressure'], 0)
        self.assertEqual(read({'day': 8, 'days': 30, 'mode': 'final'}, hist, 'me'), NEUTRAL)
        os.environ['NIGHTSHIFT_POSTURE'] = 'final'
        try:
            self.assertGreater(read({'day': 8, 'days': 30, 'mode': 'final'}, hist, 'me')['pressure'], 0)
        finally:
            del os.environ['NIGHTSHIFT_POSTURE']
