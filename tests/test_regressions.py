import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from app import app, clean_symbol, format_large
from model import ResidualOptionANN
from src.pricing import black_scholes_call, black_scholes_put, bs_greeks, implied_volatility


class PricingEdgeTests(unittest.TestCase):
    def test_expiry_and_zero_volatility(self):
        for price in (black_scholes_call, black_scholes_put):
            self.assertEqual(price(100, 100, 0, .05, .2), 0)
        self.assertAlmostEqual(black_scholes_call(100, 100, 1, .05, 0), 100 - 100 * np.exp(-.05))
        self.assertEqual(black_scholes_put(100, 100, 1, .05, 0), 0)

    def test_vectorized_expiry(self):
        np.testing.assert_allclose(black_scholes_call([90, 100, 110], 100, 0, .05, .2), [0, 0, 10])

    def test_invalid_inputs(self):
        for args in [(0, 100, 1, .05, .2), (100, 100, -1, .05, .2), (100, 100, 1, .05, -.2)]:
            with self.assertRaises(ValueError):
                black_scholes_call(*args)
        with self.assertRaises(ValueError):
            bs_greeks(100, 100, 0, .05, .2)
        with self.assertRaises(ValueError):
            implied_volatility(10, 100, 100, 1, .05, 'typo')
        self.assertTrue(np.isnan(implied_volatility(0, 100, 100, 0, .05)))

    def test_greeks_accept_lists(self):
        self.assertEqual(bs_greeks([100, 110], 100, [1, 2], .05, .2)['delta'].shape, (2,))

    def test_residual_price_is_one_per_contract(self):
        model = ResidualOptionANN()
        x = torch.zeros(3, 5)
        baseline = torch.tensor([10., 20., 30.])
        result = model.predict_price(x, baseline)
        self.assertEqual(result.shape, (3, 1))
        torch.testing.assert_close(result, torch.clamp(model(x) + baseline[:, None], min=0))


class WebTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_homepage(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Generate forecast', response.data)

    def test_options_empty_state_and_file_access(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(app.config, OPTION_RESULTS_DIR=directory):
            response = self.client.get('/options')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'No comparison run available', response.data)
            self.assertEqual(self.client.get('/options/charts/ann_direct.pth').status_code, 404)
            self.assertEqual(self.client.get('/options/charts/loss_curves.png').status_code, 404)

    def test_options_render_saved_metrics_and_chart(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(app.config, OPTION_RESULTS_DIR=directory):
            root = Path(directory)
            (root / 'overall_metrics.csv').write_text('Model,RMSE,MAE,R2\nBlack-Scholes,0.4,0.3,0.99\n')
            (root / 'loss_curves.png').write_bytes(b'example image')
            response = self.client.get('/options')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'0.4000', response.data)
            self.assertIn(b'/options/charts/loss_curves.png', response.data)
            with self.client.get('/options/charts/loss_curves.png') as chart:
                self.assertEqual(chart.status_code, 200)
                self.assertEqual(chart.mimetype, 'image/png')

    def test_invalid_requests_do_not_train(self):
        with patch('app.run_prediction') as predict:
            for body in [None, [], [1], 'TCS', {}, {'symbol': None}, {'symbol': 42}, {'symbol': ''}, {'symbol': 'AAPL.US'}]:
                response = self.client.post('/predict', json=body)
                self.assertEqual(response.status_code, 400, body)
            predict.assert_not_called()

    def test_prediction_route(self):
        with patch('app.run_prediction', return_value={'symbol': 'TCS.NS'}) as predict:
            response = self.client.post('/predict', json={'symbol': ' tcs '})
            self.assertEqual(response.status_code, 200)
            predict.assert_called_once_with('TCS.NS')

    def test_formatting(self):
        self.assertEqual(clean_symbol('m&m.ns'), 'M&M.NS')
        self.assertEqual(format_large(1e7), '₹1.00 Cr.')
        self.assertEqual(format_large(1e12), '₹1.00 L Cr.')
        self.assertEqual(format_large(float('inf')), 'N/A')

    def test_notebooks_parse_and_compile(self):
        for path in Path('.').glob('*.ipynb'):
            for i, cell in enumerate(json.loads(path.read_text(encoding='utf-8'))['cells']):
                if cell['cell_type'] == 'code':
                    compile(''.join(cell['source']), f'{path}:{i}', 'exec')


if __name__ == '__main__':
    unittest.main()
