"""
Option Data Generation and Loading Module.
Supports:
1. Synthetic market options data with realistic volatility smile/skew and market noise.
2. Sequence feature generation for LSTM temporal modeling.
3. Live option chain downloading via yfinance.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from src.pricing import black_scholes_call, black_scholes_put


def generate_market_options_data(
    n_samples: int = 10000,
    spot_price: float = 100.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic options dataset capturing the empirical volatility smile/skew.
    
    In real markets, out-of-the-money (OTM) puts and calls exhibit higher implied volatility
    than at-the-money (ATM) options (volatility smile/skew).
    
    The market price C_market is generated under dynamic market implied volatility + market microstructure noise.
    The benchmark C_BS is evaluated using static historical volatility (causing the classic BS error).
    """
    np.random.seed(seed)
    
    # 1. Underlying spot price with small variations around spot_price
    S = np.random.uniform(spot_price * 0.90, spot_price * 1.10, n_samples)
    
    # 2. Strike price K such that moneyness S/K spans [0.80, 1.20] (OTM, ATM, ITM)
    moneyness = np.random.uniform(0.80, 1.20, n_samples)
    K = S / moneyness
    
    # 3. Time to maturity T (between 5 days and 120 days, annualized)
    days_to_expiry = np.random.randint(5, 121, n_samples)
    T = days_to_expiry / 365.0
    
    # 4. Risk-free interest rate r (between 1.5% and 5.0%)
    r = np.random.uniform(0.015, 0.050, n_samples)
    
    # 5. Base historical volatility (constant BS assumption, ~20% standard deviation)
    sigma_hist = np.random.uniform(0.15, 0.25, n_samples)
    
    # 6. Empirical market volatility with smile & skew:
    # sigma_market = sigma_hist + skew*(1 - S/K) + curvature*(S/K - 1)^2 + term_structure*(1/sqrt(T))
    skew = 0.12 * (1.0 - moneyness)  # Higher IV for OTM puts (S/K < 1)
    smile = 0.25 * (moneyness - 1.0) ** 2  # Volatility smile
    term_effect = 0.015 / np.sqrt(T)  # Short-term options experience higher implied vol spikes
    sigma_market = np.clip(sigma_hist + skew + smile + term_effect, 0.08, 0.70)
    
    # 7. Generate true Market Call & Put prices with subtle microstructure noise
    c_market_clean = black_scholes_call(S, K, T, r, sigma_market)
    noise_call = np.random.normal(0.0, 0.02 * (1.0 + c_market_clean * 0.05), n_samples)
    # Market call price cannot violate intrinsic value
    intrinsic_call = np.maximum(0.0, S - K * np.exp(-r * T))
    c_market = np.maximum(intrinsic_call, c_market_clean + noise_call)
    
    p_market_clean = black_scholes_put(S, K, T, r, sigma_market)
    noise_put = np.random.normal(0.0, 0.02 * (1.0 + p_market_clean * 0.05), n_samples)
    intrinsic_put = np.maximum(0.0, K * np.exp(-r * T) - S)
    p_market = np.maximum(intrinsic_put, p_market_clean + noise_put)
    
    # 8. Classical Black-Scholes baseline price using static historical volatility
    c_bs = black_scholes_call(S, K, T, r, sigma_hist)
    p_bs = black_scholes_put(S, K, T, r, sigma_hist)
    
    # 9. Calculate Black-Scholes pricing errors (residuals)
    call_error = c_market - c_bs
    put_error = p_market - p_bs
    
    # Moneyness Category (matching the research paper):
    # OTM: S/K < 0.97, ATM: 0.97 <= S/K < 1.03, ITM: S/K >= 1.03
    moneyness_cat = np.where(moneyness < 0.97, 'OTM', np.where(moneyness < 1.03, 'ATM', 'ITM'))
    
    df = pd.DataFrame({
        'S': S,
        'K': K,
        'moneyness': moneyness,
        'T': T,
        'days': days_to_expiry,
        'r': r,
        'sigma_hist': sigma_hist,
        'sigma_market': sigma_market,
        'C_market': c_market,
        'C_bs': c_bs,
        'C_error': call_error,
        'P_market': p_market,
        'P_bs': p_bs,
        'P_error': put_error,
        'moneyness_cat': moneyness_cat
    })
    return df


def generate_time_series_option_sequences(
    df: pd.DataFrame,
    seq_len: int = 15,
    seed: int = 42
):
    """
    Creates temporal historical sequences for LSTM training.
    For each option contract, synthesizes a rolling window of recent underlying prices
    and rolling volatilities leading up to the current pricing day.
    
    Returns:
    - X_seq: (n_samples, seq_len, n_features) sequence tensor
    - X_static: (n_samples, n_static_features) static contract features [S/K, T, r, sigma_hist, C_bs/K]
    - y_market: (n_samples, 1) market prices (or scaled)
    - y_error: (n_samples, 1) pricing errors C_market - C_bs
    """
    np.random.seed(seed)
    n = len(df)
    
    # Sequence features: [relative price walk, rolling vol]
    # Simulate geometric Brownian motion lookback leading to current S
    dt = 1.0 / 252.0
    sequences = []
    
    for idx in range(n):
        s_current = df['S'].iloc[idx]
        sigma = df['sigma_hist'].iloc[idx]
        
        # Backward simulation of past stock prices leading to s_current
        bm_steps = np.random.normal(0, np.sqrt(dt), seq_len)
        log_ret = -0.5 * (sigma ** 2) * dt + sigma * bm_steps
        # Cumulative product backwards from 1.0
        cum_ret = np.exp(np.cumsum(log_ret[::-1]))[::-1]
        past_prices = s_current * cum_ret
        past_returns = np.diff(np.log(past_prices), prepend=np.log(past_prices[0]))
        
        # Sequence: shape (seq_len, 2)
        seq = np.column_stack([past_prices / df['K'].iloc[idx], past_returns])
        sequences.append(seq)
        
    X_seq = np.array(sequences, dtype=np.float32)
    
    # Static features: [S/K, T, r, sigma_hist, C_bs / K]
    X_static = np.column_stack([
        df['moneyness'].values,
        df['T'].values,
        df['r'].values,
        df['sigma_hist'].values,
        (df['C_bs'] / df['K']).values
    ]).astype(np.float32)
    
    y_market = df['C_market'].values.reshape(-1, 1).astype(np.float32)
    y_error = df['C_error'].values.reshape(-1, 1).astype(np.float32)
    
    return X_seq, X_static, y_market, y_error
