import unittest
from bot import Runner, STAFF_TRIES
from complaints import Complaints


class Client:
    def __init__(self, statuses): self.statuses, self.posts = list(statuses), 0
    def call(self, method, path, body=None, timeout=30):
        self.posts += 1
        return (self.statuses.pop(0) if self.statuses else 200), {}


class Brain:
    diagnostics = {}
    def crew_plan(self, state, order):
        return {'change': True, 'headcount': 3, 'wage': 6., 'current': (2, 6.), 'needed': 3, 'prob_enough_staff': 1.,
                'expected_daily_wages': 18., 'poach_rate_per_exposed_rival': 0., 'rivals_exposed': 0,
                'pool_median_ask': None, 'unemployed': 0}
    class crew: checks = []


def run(statuses, calls):
    runner = Runner(Client(statuses), Brain())
    runner.accepted_order = {'meals': 100}
    state = {'mode': 'final', 'market': 1, 'order_for_day': 5, 'you': {'id': 'me'}}
    for _ in range(calls):
        runner.staff(state, '/api/v1')
    return runner.client.posts


class StaffRetryTests(unittest.TestCase):
    def test_retries_a_failure_then_stops_on_success(self):
        self.assertEqual(run([500, 200], 5), 2)

    def test_bounded(self):
        self.assertEqual(run([500] * 20, 20), STAFF_TRIES)

    def test_no_resend_after_success(self):
        self.assertEqual(run([200], 5), 1)


class HorizonTests(unittest.TestCase):
    def test_no_days_left_on_the_last_day(self):
        self.assertEqual(Complaints.days_left({'days': 30, 'order_for_day': 30}), 0)
        self.assertEqual(Complaints.days_left({'days': 30, 'order_for_day': 29}), 1)
        self.assertEqual(Complaints.days_left({'days': 30, 'order_for_day': 10}), 20)


if __name__ == '__main__':
    unittest.main()
