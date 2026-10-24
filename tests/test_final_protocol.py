"""The final's protocol, end to end against a fake server: market after market (day resets to 0), an
order every night inside the window, the final profile active, and a showcase state with you: null."""
import time
import unittest
from bot import Runner, make_brain
from sim import Market


class Stop(Exception):
    pass


class FakeFinal:
    def __init__(self, markets=3, days=4, null_tail=3):
        self.markets, self.days, self.null_tail = markets, days, null_tail
        self.number, self.posts, self.gets, self.starts, self.nulls_served = 0, [], 0, [], 0
        self.shown = time.perf_counter()   # when the bot was last handed a state
        self.market, self.state = None, None
        self.begin()

    def begin(self):
        self.number += 1
        self.market = Market('final-%d' % self.number, rivals=3, days=self.days,
                             features={'fair_choice': True, 'walk_over': True, 'salvage': True})
        self.state = dict(self.market.state(), mode='final', market=self.number, markets=self.markets)
        self.starts.append(time.perf_counter())

    def call(self, method, path, body=None, timeout=30):
        if method == 'GET':
            self.shown = time.perf_counter()
            self.gets += 1
            if self.number > self.markets:
                self.nulls_served += 1
                if self.nulls_served > self.null_tail:
                    raise Stop()
                return 200, dict(self.state, mode='showcase', you=None, version=10_000 + self.nulls_served)
            return 200, self.state
        if not path.endswith('/orders'):
            return 200, {'ok': True}              # votes, staff, complaints: not part of the order window
        self.posts.append((self.number, self.state.get('order_for_day'), body, time.perf_counter() - self.shown))
        if self.number > self.markets:
            return 403, {'error': 'not playing'}
        played = dict(self.market.play(body), mode='final', market=self.number, markets=self.markets)
        self.shown = time.perf_counter()
        if played.get('phase') == 'over':
            if self.number < self.markets:
                self.begin()                      # the next market starts at once, day back to 0
            else:
                self.number += 1                  # the final is over: showcase states with you: null follow
                self.state = played
            return 200, {'state': self.state}
        self.state = played
        return 200, {'state': played}


class FinalProtocolTests(unittest.TestCase):
    def test_every_night_of_every_market_gets_an_order_and_a_null_you_gets_none(self):
        server = FakeFinal()
        runner = Runner(server, make_brain(samples=12))
        with self.assertRaises(Stop):
            runner.play()
        real = [p for p in server.posts if p[0] <= server.markets]
        self.assertEqual(len(real), server.markets * server.days)             # one order per night, per market
        for market in range(1, server.markets + 1):
            days = [p[1] for p in real if p[0] == market]
            self.assertEqual(days, list(range(1, server.days + 1)))             # day 1 first, even after a reset
        self.assertEqual(len([p for p in server.posts if p[0] > server.markets]), 0)   # null you: nothing sent
        self.assertTrue(runner.brain.final_active)
        self.assertEqual(runner.brain.final_policy.final_started, True)
        slowest = max(p[3] for p in real)
        self.assertLess(slowest, 1.5)                                           # well inside the 2 s window

    def test_reputation_and_cash_start_fresh_each_market(self):
        server = FakeFinal(markets=2, days=3, null_tail=1)
        runner = Runner(server, make_brain(samples=8))
        with self.assertRaises(Stop):
            runner.play()
        self.assertIn(runner.brain.market[0], ('final', 'showcase'))
        self.assertEqual(runner.brain.market[1], 2)               # the brain followed the market counter


if __name__ == '__main__':
    unittest.main()
