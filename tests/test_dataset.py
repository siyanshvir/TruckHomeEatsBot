import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io

from dataset import FinalLog, calibration_rows, iter_days
from game import reputation_value, VALUES
import tools.train_final as train_final
import tools.final_eval as final_eval


def last_day(day, price=9., profit=10.):
    spots = {'oval': {'customers': 50, 'hungry': 5, 'trucks': [
        {'id': 'me', 'price': price, 'quality': 'premium', 'meals': 50, 'sold': 40},
        {'id': 'r1', 'price': 8., 'quality': 'standard', 'meals': 30, 'sold': 5}]}}
    return {'day': day, 'weather': 'clear', 'events': [], 'spots': spots,
            'you': {'status': 'played', 'spot': 'oval', 'price': price, 'quality': 'premium',
                    'meals': 50, 'sold': 40, 'profit': profit}}


def observation(market, day, price=9., profit=10., mode='live'):
    return {'mode': mode, 'market': market, 'day': day, 'last_day': last_day(day, price, profit),
            'you': {'id': 'me', 'reputation': {'oval': 50+day}}, 'trucks': []}


def write_tree(root):
    folder = Path(root)/'live-observations'
    folder.mkdir(parents=True)
    # Written out of order on purpose; a malformed and a non-dict file must be skipped.
    for market, day in [(2, 3), (1, 5), (1, 2), (2, 1), (1, 3)]:
        (folder/('live-%d-day%02d.json' % (market, day))).write_text(
            json.dumps(observation(market, day, price=float(day), profit=float(day*10))))
    (folder/'broken.json').write_text('{"mode": "live", ')
    (folder/'list.json').write_text('[1, 2]')
    (folder/'nolast.json').write_text('{"mode": "live", "market": 1}')
    forecast = {'mode': 'live', 'market': 1, 'day': 3, 'input_forecast': {
        'expected_customers': {'oval': 99}, 'weather': {'clear': 1.}, 'events': []}}
    (Path(root)/'forecasts-v2.jsonl').write_text(json.dumps(forecast)+'\nnot json\n')


class DatasetTests(unittest.TestCase):
    def test_order_and_no_future_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            write_tree(folder)
            days = list(iter_days(folder))
            self.assertEqual([(d['market'], d['day']) for d in days],
                             [(1, 2), (1, 3), (1, 5), (2, 1), (2, 3)])
            self.assertEqual(days[1]['forecast']['expected_customers'], {'oval': 99})
            self.assertEqual(days[0]['features'], {'prior_days': 0})
            # Features of a day see only earlier days of the same market.
            self.assertEqual(days[1]['features']['prev_price'], 2.)
            self.assertEqual(days[2]['features']['prev_price'], 3.)
            self.assertEqual(days[2]['features']['prior_days'], 2)
            self.assertEqual(days[3]['features'], {'prior_days': 0})
            self.assertEqual(days[2]['you']['reputation_before'], {'oval': 53})
            self.assertEqual(days[4]['features']['prev_price'], 1.)
            self.assertEqual(days[0]['spots']['oval']['trucks'][0]['id'], 'me')
            self.assertEqual(list(iter_days(folder, include=('practice',))), [])

    def test_missing_root_and_practice_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(iter_days(Path(folder)/'nothing')), [])
            row = {'seed': 7, 'me': 'me', 'last_day': last_day(1),
                   'pre': {'day': 1, 'forecast': {'expected_customers': {'oval': 10}},
                           'trucks': [{'id': 'me', 'rep': {'oval': 61}}]}}
            (Path(folder)/'collect.jsonl').write_text('garbage\n'+json.dumps(row)+'\n')
            days = list(iter_days(folder))
            self.assertEqual([(d['mode'], d['market']) for d in days], [('practice', '7')])
            self.assertEqual(days[0]['you']['reputation_before'], {'oval': 61})

    def test_calibration_rows_use_value_ceiling(self):
        base = {'order': {'spot': 'oval', 'quality': 'premium', 'price': 8., 'meals': 60},
                'expected_sales': 50., 'actual_sales': 40., 'sellout': False}
        rows = calibration_rows([base, dict(base, reputation=100), {'bad': 1}])
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]['value_ceiling'], reputation_value('premium', 50, VALUES))
        self.assertAlmostEqual(rows[0]['price_ratio'], 8./rows[0]['value_ceiling'])
        self.assertAlmostEqual(rows[1]['value_ceiling'], reputation_value('premium', 100, VALUES))
        full = calibration_rows([dict(base, actual_sales=60.)])
        self.assertTrue(full[0]['sellout'])

    def test_final_log_append_dedupe_and_truncated_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'sub'/'final-days.jsonl'
            log = FinalLog(path)
            self.assertTrue(log.append({'mode': 'final', 'market': 1, 'day': 1, 'x': 1}))
            self.assertFalse(log.append({'mode': 'final', 'market': 1, 'day': 1, 'x': 2}))
            with path.open('a') as out:
                out.write('{"mode": "final", "market": 1, "da')  # crash mid-write
            reopened = FinalLog(path)
            self.assertTrue(reopened.append({'mode': 'final', 'market': 1, 'day': 2}))
            self.assertFalse(FinalLog(path).append({'mode': 'final', 'market': 1, 'day': 1}))
            self.assertEqual([r['day'] for r in FinalLog(path).rows()], [1, 2])

    def test_final_mode_days_are_read(self):
        with tempfile.TemporaryDirectory() as folder:
            record = dict(last_day(4), mode='final', market='f1', forecast={'expected_customers': {'oval': 5}},
                          you=dict(last_day(4)['you'], id='me', reputation={'oval': 70}))
            FinalLog(Path(folder)/'final-days.jsonl').append(record)
            days = list(iter_days(folder, include=('final',)))
            self.assertEqual([(d['mode'], d['market'], d['day']) for d in days], [('final', 'f1', 4)])
            self.assertEqual(days[0]['forecast']['expected_customers'], {'oval': 5})


class TrainFinalTests(unittest.TestCase):
    def write_eval(self, root):
        rows = []
        for day in range(1, 13):
            sellout = day % 4 == 0
            rows.append({'mode': 'live', 'market': 1, 'day': day,
                         'order': {'spot': 'oval', 'quality': 'premium', 'price': 8.+day/4., 'meals': 50},
                         'expected_sales': 40., 'actual_sales': 50. if sellout else 30., 'sellout': sellout})
        (Path(root)/'evaluation-v2.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')

    def test_priors_valid_deterministic_and_prior_json_untouched(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_tree(root)
            self.write_eval(root)
            output = root/'model'/'final_priors.json'
            guard = root/'model'/'prior.json'
            guard.parent.mkdir(parents=True)
            guard.write_text('sentinel')
            with contextlib.redirect_stdout(io.StringIO()):
                train_final.main(['--data-root', str(root), '--output', str(output)])
                first = output.read_bytes()
                train_final.main(['--data-root', str(root), '--output', str(output)])
            self.assertEqual(first, output.read_bytes())
            priors = json.loads(first)
            self.assertEqual(priors['version'], 1)
            glob = priors['calibration']['global']
            self.assertEqual(glob['n'], 9)
            self.assertEqual(glob['censored'], 3)
            self.assertAlmostEqual(glob['mean'], .75)  # sellouts excluded from the mean
            self.assertEqual(priors['controller']['risk']['1.0'], {'mean': 0, 'var': 1, 'n': 0})
            self.assertEqual(priors['meta']['days'], 5)
            self.assertEqual(priors['meta']['source_modes'], {'live': 5})
            self.assertEqual(guard.read_text(), 'sentinel')
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                train_final.main(['--data-root', str(root), '--output', str(guard)])
            self.assertEqual(guard.read_text(), 'sentinel')

    def test_empty_data_root_still_produces_neutral_priors(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'p.json'
            with contextlib.redirect_stdout(io.StringIO()):
                train_final.main(['--data-root', folder, '--output', str(output)])
            self.assertEqual(json.loads(output.read_text())['calibration']['global']['n'], 0)


class FinalEvalTests(unittest.TestCase):
    def test_baseline_vs_unrestricted_and_final_skipped_when_absent(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'eval.json'
            with patch.dict('sys.modules', {'final_mode': None}), contextlib.redirect_stdout(io.StringIO()) as out:
                report = final_eval.main(['--seeds', 'a', '--rivals', '3', '--days', '3', '--styles', 'follower', 'mixed',
                                          '--policies', 'baseline', 'unrestricted', 'final', '--samples', '4',
                                          '--features', 'salvage', '--output', str(output)])
            self.assertEqual(report['skipped_policies'], ['final'])
            self.assertEqual(report['policies'], ['baseline', 'unrestricted'])
            self.assertEqual(len(report['results']), 4)
            self.assertIn('NOTICE', out.getvalue())
            self.assertNotIn('regressions', report)
            self.assertEqual(json.loads(output.read_text())['summary']['baseline']['markets'], 2)

    def test_markets_limit_and_regression_report_with_stub_final(self):
        import types
        from bot import make_brain
        stub = types.SimpleNamespace(make_final_brain=lambda: make_brain(samples=4))
        with patch.dict('sys.modules', {'final_mode': stub}):
            report = final_eval.evaluate(['a'], [3], 3, ['follower', 'mixed'], ['baseline', 'final'],
                                         markets=1, samples=4)
        self.assertEqual(len(report['results']), 2)
        self.assertEqual(len(report['regressions']['markets']), 1)
        self.assertEqual(report['regressions']['markets'][0]['profit_delta'], 0)
