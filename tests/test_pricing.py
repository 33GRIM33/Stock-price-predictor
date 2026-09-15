"""
Unit tests for Black-Scholes pricing formulas and Greeks.
"""

import unittest
import numpy as np
from src.pricing import black_scholes_call, black_scholes_put, bs_greeks, implied_volatility


class TestOptionPricing(unittest.TestCase):

    def test_bs_call_known_value(self):
        # S=100, K=100, T=1, r=0.05, sigma=0.20
        # Expected European call price is approx 10.4506
        call_price = black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.20)
        self.assertAlmostEqual(call_price, 10.4506, places=3)

    def test_bs_put_call_parity(self):
        # C - P = S - K * exp(-r * T)
        S, K, T, r, sigma = 105.0, 100.0, 0.5, 0.03, 0.25
        call = black_scholes_call(S, K, T, r, sigma)
        put = black_scholes_put(S, K, T, r, sigma)
        parity_lhs = call - put
        parity_rhs = S - K * np.exp(-r * T)
        self.assertAlmostEqual(parity_lhs, parity_rhs, places=5)

    def test_greeks_bounds(self):
        greeks = bs_greeks(100.0, 100.0, 1.0, 0.05, 0.20, option_type='call')
        # Delta for ATM call should be ~0.5 to 0.65
        self.assertTrue(0.5 <= greeks['delta'] <= 0.7)
        # Vega should be positive
        self.assertTrue(greeks['vega'] > 0)
        # Gamma should be positive
        self.assertTrue(greeks['gamma'] > 0)

    def test_implied_volatility_inversion(self):
        true_sigma = 0.25
        call = black_scholes_call(100.0, 100.0, 0.5, 0.03, true_sigma)
        iv = implied_volatility(call, 100.0, 100.0, 0.5, 0.03, option_type='call')
        self.assertAlmostEqual(iv, true_sigma, places=3)


if __name__ == '__main__':
    unittest.main()
