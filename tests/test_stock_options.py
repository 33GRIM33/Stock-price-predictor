import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app import app
from src.pricing import black_scholes_call
from src.stock_options import (clean_history, make_dataset, date_partitions,
                               compare_stock_options, read_contracts)


def history_fixture(scale=1.):
    rng = np.random.default_rng(7)
    close = 500 * scale * np.exp(np.cumsum(rng.normal(.0002, .012, 260)))
    return pd.DataFrame({'Close': close}, index=pd.bdate_range('2024-01-01', periods=260))


def contract_fixture(history):
    rows = []
    for date, price in history.Close.iloc[-30:].items():
        vol = np.diff(np.log(history.Close.loc[:date].iloc[-61:])).std(ddof=1) * np.sqrt(252)
        for ratio in [.95, 1., 1.05]:
            strike = price * ratio
            rows.append(dict(symbol='SBIN', date=date.strftime('%Y-%m-%d'),
                             expiry=(date + pd.Timedelta(days=30)).strftime('%Y-%m-%d'),
                             spot=price, strike=strike,
                             call_price=float(black_scholes_call(price, strike, 30/365, .05, vol))))
    return pd.DataFrame(rows)


class StockOptionDataTests(unittest.TestCase):
    def test_past_only_features_and_date_separation(self):
        close = clean_history(history_fixture())
        original, sequences = make_dataset(close, max_dates=30, paths=64)
        changed = close.copy()
        changed.iloc[-10:] *= 1.3
        modified, modified_seq = make_dataset(changed, max_dates=30, paths=64)
        early = original.date < close.index[-10]
        np.testing.assert_allclose(original.loc[early, 'target'], modified.loc[early, 'target'])
        np.testing.assert_allclose(sequences[early], modified_seq[early])
        tr, va, te = date_partitions(original)
        self.assertLess(original.iloc[tr].date.max(), original.iloc[va].date.min())
        self.assertLess(original.iloc[va].date.max(), original.iloc[te].date.min())
        self.assertEqual(len(tr) + len(va) + len(te), len(original))

    def test_stock_price_scale_changes_contracts_and_reference(self):
        first, _ = make_dataset(clean_history(history_fixture()), max_dates=20, paths=64)
        second, _ = make_dataset(clean_history(history_fixture(2.)), max_dates=20, paths=64)
        np.testing.assert_allclose(second.target, first.target * 2, rtol=1e-5)
        np.testing.assert_allclose(second.strike, first.strike * 2)

    def test_csv_validation(self):
        data = contract_fixture(history_fixture())
        parsed = read_contracts(io.StringIO(data.to_csv(index=False)), 'SBIN.NS')
        self.assertEqual(len(parsed), 90)
        for bad in [data.assign(symbol='TCS'), data.assign(call_price=-1),
                    data.assign(date='2024-12-01'), data.assign(expiry='bad'),
                    pd.concat([data, data.iloc[:1]])]:
            with self.assertRaises(ValueError):
                read_contracts(io.StringIO(bad.to_csv(index=False)), 'SBIN.NS')

    def test_observed_targets_are_not_replaced_and_spot_is_from_csv(self):
        history = history_fixture()
        source = contract_fixture(history)
        parsed = read_contracts(io.StringIO(source.to_csv(index=False)), 'SBIN.NS')
        frame, _ = make_dataset(clean_history(history), contracts=parsed)
        np.testing.assert_allclose(frame.target, parsed.call_price)
        np.testing.assert_allclose(frame.spot, parsed.spot)
        parsed['call_price'] = parsed.spot * 2
        with self.assertRaisesRegex(ValueError, 'bounds'):
            make_dataset(clean_history(history), contracts=parsed)

    def test_complete_simulated_comparison(self):
        result = compare_stock_options('SBIN.NS', history_fixture(), epochs=1)
        self.assertEqual(result['mode'], 'simulation')
        self.assertEqual(len(result['metrics']), 4)
        self.assertEqual(len(result['contracts']), result['split']['test']['rows'])
        self.assertEqual(result['symbol'], 'SBIN.NS')
        for row in result['contracts']:
            self.assertGreater(row['mc_se'], 0)
            self.assertTrue(all(np.isfinite(v) and 0 <= v <= row['spot'] for v in row['predictions'].values()))


class StockOptionRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_invalid_requests_never_download(self):
        with patch('app.yf.download') as download:
            for body in [None, [], ['x'], {}, {'symbol':'SBIN', 'rate':'nan'},
                         {'symbol':'../SBIN'}, {'symbol':'SBIN', 'mode':'live'},
                         {'symbol':'SBIN', 'mode':'observed'}, {'symbol':'SBIN', 'dividend':-1}]:
                self.assertEqual(self.client.post('/api/stock-options', json=body).status_code, 400)
            download.assert_not_called()

    def test_observed_upload_runs_all_models(self):
        history = history_fixture()
        csv = contract_fixture(history).to_csv(index=False).encode()
        with patch('app.yf.download', return_value=history) as download, patch.dict(app.config, STOCK_OPTION_EPOCHS=1):
            response = self.client.post('/api/stock-options', data={
                'symbol':'sbin', 'mode':'observed', 'rate':'0.05', 'dividend':'0',
                'contracts':(io.BytesIO(csv), 'calls.csv')})
        self.assertEqual(response.status_code, 200, response.get_json())
        result = response.get_json()
        self.assertEqual(result['mode'], 'observed')
        self.assertEqual(result['symbol'], 'SBIN.NS')
        self.assertEqual(len(result['metrics']), 4)
        self.assertLess(result['metrics'][0]['rmse'], 1e-8)
        self.assertIsNotNone(result['contracts'][0]['expiry'])
        self.assertEqual(download.call_args.args[0], 'SBIN.NS')
        self.assertTrue(download.call_args.kwargs['auto_adjust'])

    def test_bad_or_missing_history_is_visible_error(self):
        with patch('app.yf.download', return_value=pd.DataFrame()):
            response = self.client.post('/api/stock-options', json={'symbol':'SBIN'})
        self.assertEqual(response.status_code, 400)
        self.assertIn('history', response.get_json()['error'])

    def test_oversized_upload_and_ui(self):
        with patch.dict(app.config, MAX_CONTENT_LENGTH=100):
            response = self.client.post('/api/stock-options', data={'symbol':'SBIN', 'padding':'x'*200})
        self.assertEqual(response.status_code, 413)
        self.assertIn('2 MB', response.get_json()['error'])
        page = self.client.get('/').get_data(as_text=True)
        self.assertIn('Which model performed best?', page)
        self.assertNotIn('id="option-mode"', page)
        self.assertNotIn('id="rate"', page)
        self.assertNotIn('id="dividend"', page)
        with self.client.get('/static/option_contracts_template.csv') as template:
            self.assertEqual(template.status_code, 200)


if __name__ == '__main__':
    unittest.main()
