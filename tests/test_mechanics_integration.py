import copy
import json
import unittest
import itertools
from bot import make_brain
from game import SPOTS, Rules
from mechanics import food_profit
from sim import Market


class MechanicsIntegrationTests(unittest.TestCase):
    def test_all_feature_combinations_keep_planner_and_log_in_agreement(self):
        names = ('fair_choice', 'walk_over', 'salvage')
        for values in itertools.product((False, True), repeat=3):
            flags = dict(zip(names, values))
            state = Market('all-feature-combos', rivals=3, days=3, features=flags).state()
            brain = make_brain(samples=4)
            brain.decide(state)
            rows = brain.forecast_record['scenarios']
            chosen = brain.diagnostics['chosen']
            self.assertAlmostEqual(sum(r['sales'] for r in rows)/len(rows), chosen['sales'])
            self.assertAlmostEqual(sum(r['profit'] for r in rows)/len(rows), chosen['profit'])
            self.assertAlmostEqual(sum(max(0., -r['profit']) for r in rows)/len(rows), chosen['loss'])

    def test_salvage_is_exact_quarter_of_unsold_ingredients(self):
        self.assertEqual(food_profit(15, 84, 7.88, 5.5, False), -343.8)
        self.assertAlmostEqual(food_profit(15, 84, 7.88, 5.5, True), -248.925)
        self.assertEqual(food_profit(60, 60, 8.5, 5.5, True), 180)

    def test_combined_features_keep_constraints_and_ledger_profit_consistent(self):
        state = Market('feature-integration', rivals=3, days=3,
                       features={'fair_choice': True, 'walk_over': True, 'salvage': True}).state()
        brain = make_brain(samples=4)
        order = brain.decide(state)
        Rules.read(state).validate(order)
        allowed = sorted(SPOTS, key=lambda s: state['forecast']['expected_customers'][s])[:3]
        self.assertIn(order['spot'], allowed)
        self.assertLessEqual(order['price'], brain.low_price_caps[order['quality']])
        self.assertEqual(brain.learner.active_features, (True, True, True))
        rows = brain.forecast_record['scenarios']
        self.assertAlmostEqual(sum(r['profit'] for r in rows)/len(rows), brain.diagnostics['chosen']['profit'])
        self.assertEqual(brain.forecast_record['features'], state['features'])
        self.assertIsNone(brain.diagnostics.get('forecast_capture_error'))

    def test_new_flags_change_rule_fingerprint_and_survive_checkpoint(self):
        state = Market('flag-identity', rivals=3, days=3).state()
        old, new = make_brain(samples=4), make_brain(samples=4)
        old.decide(state)
        state['features'] = {'salvage': True}
        new.decide(state)
        self.assertNotEqual(old.forecast_record['rules_digest'], new.forecast_record['rules_digest'])
        restored = make_brain(samples=4)
        restored.restore(json.loads(json.dumps(new.dump())))
        self.assertEqual(restored.learner.active_features, (False, False, True))

    def test_future_flags_do_not_relabel_completed_day_learning(self):
        market = Market('future-toggle', rivals=3, days=3)
        brain = make_brain(samples=4)
        order = brain.decide(market.state())
        state = market.play(order)
        state['last_day'].pop('features')
        state['features'] = {'fair_choice': True, 'walk_over': True, 'salvage': True}
        brain.observe(state)
        self.assertEqual(brain.learner.active_features, (False, False, False))
        legacy_rows = brain.learner.rows
        brain.decide(state)
        self.assertEqual(brain.learner.active_features, (True, True, True))
        self.assertEqual(brain.learner.rows, 0)
        brain.learner.activate({})
        self.assertEqual(brain.learner.rows, legacy_rows)
