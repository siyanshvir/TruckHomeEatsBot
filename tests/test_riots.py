import unittest
from riots import riot_cost, warnings


def state(events, riots=True):
    return {'features': {'riots': riots}, 'forecast': {'events': events}}


class RiotTests(unittest.TestCase):
    def test_no_warning_or_rule_off_costs_nothing(self):
        self.assertEqual(riot_cost(state([]), 'highst'), 0)
        self.assertEqual(riot_cost(state([{'name': 'Riot expected on High Street'}], riots=False), 'highst'), 0)

    def test_michigan_warning_prices_high_street_far_above_distant_spots(self):
        s = state([{'key': 'game', 'name': 'The Game vs Michigan', 'effect': 'fans may riot if Ohio State loses'}])
        near, mid, far = riot_cost(s, 'highst'), riot_cost(s, 'northres'), riot_cost(s, 'stadium')
        self.assertGreater(near, mid)
        self.assertGreater(mid, far)
        self.assertGreater(near, 20)

    def test_title_warning_centres_on_the_stadium(self):
        s = state([{'name': 'National Championship', 'effect': 'a title could start a riot', 'spot': 'stadium'}])
        self.assertEqual(warnings(s['forecast'])[0][0], 'title')
        self.assertGreater(riot_cost(s, 'stadium'), riot_cost(s, 'thompson'))
        self.assertGreater(riot_cost(s, 'thompson'), 0)

    def test_malformed_forecast_is_safe(self):
        self.assertEqual(riot_cost({'features': {'riots': True}, 'forecast': None}, 'oval'), 0)
        self.assertEqual(riot_cost({'features': {'riots': True}, 'forecast': {'events': [None, 3, 'riot?']}}, 'nowhere'), 0)


if __name__ == '__main__':
    unittest.main()


class LawRiotMultiplierTests(unittest.TestCase):
    def _state(self, key, riot):
        s = state([{'key': 'game', 'name': 'The Game vs Michigan', 'effect': 'fans may riot if Ohio State loses'}])
        s['politics'] = {'laws': [{'key': key, 'params': {'game_crowd': {'other': 1.}, 'riot': riot}}]}
        return s

    def test_crackdown_removes_riot_cost_and_tailgate_raises_it(self):
        none = riot_cost(state([{'name': 'riot risk', 'effect': 'riot after the game'}]), 'highst')
        crackdown = riot_cost(self._state('crackdown', 0), 'highst')
        tailgate = riot_cost(self._state('tailgate', 1.5), 'highst')
        self.assertEqual(crackdown, 0)
        self.assertGreater(tailgate, riot_cost(self._state('x', 1.0), 'highst'))
        self.assertGreater(none, 0)
