import json
from pathlib import Path
import tempfile
import unittest
import contextlib
import io
from unittest.mock import patch
from bot import make_brain, Runner
from forecast_log import append_unique, quantiles
from sim import Market


class ForecastLogTests(unittest.TestCase):
    def test_evidence_failure_does_not_change_the_order(self):
        state = Market('evidence-failure', rivals=3, days=3).state()
        expected = make_brain(samples=4).decide(state)
        brain = make_brain(samples=4)
        with patch('strategy.capture', side_effect=ValueError('bad evidence')):
            actual = brain.decide(state)
        self.assertEqual(actual, expected)
        self.assertIsNone(brain.forecast_record)
        self.assertEqual(brain.diagnostics['forecast_capture_error'], 'ValueError')
    def test_frozen_scenarios_match_chosen_forecast_and_do_not_mutate(self):
        market = Market('ledger', rivals=3, days=3)
        brain = make_brain(samples=4)
        order = brain.decide(market.state())
        frozen = json.dumps(brain.forecast_record, sort_keys=True)
        rows = brain.forecast_record['scenarios']
        self.assertAlmostEqual(sum(x['sales'] for x in rows)/len(rows), brain.diagnostics['chosen']['sales'])
        self.assertAlmostEqual(sum(x['profit'] for x in rows)/len(rows), brain.diagnostics['chosen']['profit'])
        self.assertTrue(brain.forecast_record['named_rival_scenarios_available'])
        brain.observe(market.play(order))
        self.assertEqual(json.dumps(brain.forecast_record, sort_keys=True), frozen)
        self.assertEqual(brain.prediction_checks[-1]['forecast_id'], brain.forecast_record['forecast_id'])

    def test_append_only_predictions_and_evaluations_are_joined_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            market = Market('ledger-join', rivals=3, days=3)
            brain = make_brain(samples=4)
            state = market.state()
            order = brain.decide(state)
            runner = Runner(None, brain, log=root/'decisions.jsonl')
            runner.record(state, order)
            runner.record(state, order)
            result = market.play(order)
            brain.observe(result)
            runner.record_evaluation(result)
            runner.record_evaluation(result)
            predictions = [json.loads(x) for x in (root/'practice-forecasts-v2.jsonl').read_text().splitlines()]
            evaluations = [json.loads(x) for x in (root/'practice-evaluation-v2.jsonl').read_text().splitlines()]
            self.assertEqual(len(predictions), 1)
            self.assertEqual(len(evaluations), 1)
            self.assertEqual(predictions[0]['forecast_id'], evaluations[0]['forecast_id'])
            self.assertTrue(predictions[0]['accepted'])
            self.assertNotIn('actual_result', predictions[0])

    def test_quantiles_and_truncated_append_record(self):
        self.assertEqual(quantiles([0, 10, 20])['p50'], 10)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'test.jsonl'
            path.write_text('{broken')
            self.assertTrue(append_unique(path, {'forecast_id': 'x'}))
            self.assertFalse(append_unique(path, {'forecast_id': 'x'}))
            self.assertEqual(json.loads(path.read_text().splitlines()[-1]), {'forecast_id': 'x'})

    def test_rule_changes_are_labeled_separately_from_same_market(self):
        state = Market('rule-version', rivals=3, days=3).state()
        first = make_brain(samples=4)
        first.decide(state)
        state['rules']['budget'] = 400
        second = make_brain(samples=4)
        second.decide(state)
        self.assertNotEqual(first.forecast_record['rules_digest'], second.forecast_record['rules_digest'])
        self.assertNotEqual(first.forecast_record['regime_id'], second.forecast_record['regime_id'])

    def test_full_practice_loop_records_every_result_not_only_final(self):
        class Client:
            def __init__(self): self.market = Market('ledger-full-loop', rivals=3, days=3)
            def call(self, method, path, body=None, **kwargs):
                if method == 'GET':
                    return 200, self.market.state()
                return 200, {'state': self.market.play(body)}
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            runner = Runner(Client(), make_brain(samples=4), log=root/'decisions.jsonl')
            runner.play('/api/v1/practice/test')
            rows = [json.loads(x) for x in (root/'practice-evaluation-v2.jsonl').read_text().splitlines()]
            self.assertEqual([r['day'] for r in rows], [1, 2, 3])

    def test_distinct_sampling_configs_do_not_share_forecast_identity(self):
        state = Market('config-identity', rivals=3, days=3).state()
        state['forecast']['expected_customers'] = dict.fromkeys(state['rules']['spots'], 0)
        first, second = make_brain(samples=4), make_brain(samples=8)
        self.assertEqual(first.decide(state), second.decide(state))
        self.assertNotEqual(first.forecast_record['forecast_id'], second.forecast_record['forecast_id'])


class LedgerTailTests(unittest.TestCase):
    def test_duplicates_are_caught_from_the_tail_without_parsing_the_whole_file(self):
        import json, tempfile
        from pathlib import Path
        import forecast_log
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ledger.jsonl'
            for i in range(40):
                self.assertTrue(forecast_log.append_unique(path, {'forecast_id': 'id%d' % i, 'pad': 'x' * 50}))
            self.assertFalse(forecast_log.append_unique(path, {'forecast_id': 'id39'}))   # latest row: caught
            self.assertEqual(len(path.read_text().splitlines()), 40)
            self.assertTrue(forecast_log.append_unique(path, {'forecast_id': 'brand-new'}))
