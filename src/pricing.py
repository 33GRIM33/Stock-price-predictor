"""
Classical Black-Scholes Formula, Greeks, and Implied Volatility Computation.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def _inputs(S, K, T, r, sigma):
    values = np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in (S, K, T, r, sigma)])
    S, K, T, r, sigma = values
    if any(not np.all(np.isfinite(v)) for v in values):
        raise ValueError("Pricing inputs must be finite")
    if np.any(S <= 0) or np.any(K <= 0) or np.any(T < 0) or np.any(sigma < 0):
        raise ValueError("S and K must be positive; T and sigma must be nonnegative")
    return values


def bs_d1_d2(S, K, T, r, sigma):
    """
    Computes d1 and d2 parameters of Black-Scholes model.
    Accepts scalar or numpy arrays.
    """
    S = np.maximum(np.asarray(S, dtype=float), 1e-8)
    K = np.maximum(np.asarray(K, dtype=float), 1e-8)
    T = np.maximum(np.asarray(T, dtype=float), 1e-6)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    r = np.asarray(r, dtype=float)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_call(S, K, T, r, sigma):
    """
    Analytical Black-Scholes European Call Price:
    C = S * N(d1) - K * exp(-r * T) * N(d2)
    """
    S, K, T, r, sigma = _inputs(S, K, T, r, sigma)
    d1, d2 = bs_d1_d2(S, K, T, r, sigma)
    T = np.asarray(T, dtype=float)
    K = np.asarray(K, dtype=float)
    r = np.asarray(r, dtype=float)
    S = np.asarray(S, dtype=float)

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    intrinsic = np.maximum(0.0, S - K * np.exp(-r * T))
    return np.where((T == 0) | (sigma == 0), intrinsic, np.maximum(price, intrinsic))


def black_scholes_put(S, K, T, r, sigma):
    """
    Analytical Black-Scholes European Put Price:
    P = K * exp(-r * T) * N(-d2) - S * N(-d1)
    """
    S, K, T, r, sigma = _inputs(S, K, T, r, sigma)
    d1, d2 = bs_d1_d2(S, K, T, r, sigma)
    T = np.asarray(T, dtype=float)
    K = np.asarray(K, dtype=float)
    r = np.asarray(r, dtype=float)
    S = np.asarray(S, dtype=float)

    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    intrinsic = np.maximum(0.0, K * np.exp(-r * T) - S)
    return np.where((T == 0) | (sigma == 0), intrinsic, np.maximum(price, intrinsic))


def bs_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Computes analytical Black-Scholes Greeks:
    Delta, Gamma, Vega, Theta, Rho
    """
    S, K, T, r, sigma = _inputs(S, K, T, r, sigma)
    if option_type.lower() not in ('call', 'put'):
        raise ValueError("option_type must be call or put")
    if np.any(T == 0) or np.any(sigma == 0):
        raise ValueError("Greeks require positive T and sigma")
    d1, d2 = bs_d1_d2(S, K, T, r, sigma)
    sqrt_T = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    exp_rT = np.exp(-r * T)

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * sqrt_T * pdf_d1

    if option_type.lower() == 'call':
        delta = norm.cdf(d1)
        theta = -(S * pdf_d1 * sigma) / (2 * sqrt_T) - r * K * exp_rT * norm.cdf(d2)
        rho = K * T * exp_rT * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1.0
        theta = -(S * pdf_d1 * sigma) / (2 * sqrt_T) + r * K * exp_rT * norm.cdf(-d2)
        rho = -K * T * exp_rT * norm.cdf(-d2)

    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }


def implied_volatility(market_price, S, K, T, r, option_type='call', bounds=(0.001, 4.0)):
    """
    Inverts Black-Scholes formula using Brent's root-finding method.
    Returns NaN if market price violates arbitrage bounds.
    """
    if option_type.lower() not in ('call', 'put'):
        raise ValueError("option_type must be call or put")
    if T <= 0 or not np.isfinite(market_price):
        return np.nan

    def objective(sigma_guess):
        if option_type.lower() == 'call':
            theoretical = black_scholes_call(S, K, T, r, sigma_guess)
        else:
            theoretical = black_scholes_put(S, K, T, r, sigma_guess)
        return theoretical - market_price

    try:
        f_low = objective(bounds[0])
        f_high = objective(bounds[1])
        if f_low * f_high > 0:
            return np.nan
        return brentq(objective, bounds[0], bounds[1], xtol=1e-5, maxiter=100)
    except Exception:
        return np.nan
