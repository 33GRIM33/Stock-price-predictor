import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# ─── Helpers ────────────────────────────────────────────────────────────────

def clean_symbol(raw: str) -> str:
    raw = raw.strip().upper()
    if not raw.endswith(".NS"):
        raw = raw + ".NS"
    return raw


def safe_get(d: dict, *keys, default="N/A"):
    for k in keys:
        v = d.get(k)
        if v is not None and v != "":
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                continue
            return v
    return default


def format_large(num, default="N/A"):
    if num == "N/A" or num is None:
        return "N/A"
    try:
        num = float(num)
        if np.isnan(num):
            return "N/A"
        if num >= 1e12:
            return f"₹{num/1e12:.2f}T Cr."
        if num >= 1e7:
            return f"₹{num/1e7:.2f} L Cr."
        if num >= 1e5:
            return f"₹{num/1e5:.2f} L"
        return f"₹{num:,.2f}"
    except Exception:
        return "N/A"


def prepare_sequences(scaled_data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i])
        y.append([scaled_data[i][0], scaled_data[i][3]])  # Open, Close
    return np.array(X), np.array(y)


def inverse_pred(pred_arr, scaler):
    dummy = np.zeros((len(pred_arr), 5))
    dummy[:, 0] = pred_arr[:, 0]
    dummy[:, 3] = pred_arr[:, 1]
    return scaler.inverse_transform(dummy)


# ─── Prediction Engine ───────────────────────────────────────────────────────

def run_prediction(symbol: str):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    ticker = yf.Ticker(symbol)
    df = yf.download(symbol, start="2021-01-01",
                     end=pd.Timestamp.today().strftime('%Y-%m-%d'), progress=False)

    if df.empty or len(df) < 120:
        raise ValueError(f"Not enough data for {symbol}. Try a different stock.")

    # Flatten MultiIndex if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    info = ticker.info or {}

    # ── Feature matrix ──────────────────────────────────────────────────────
    stock_data = df[["Open", "High", "Low", "Close", "Volume"]].values

    split = int(0.8 * len(stock_data))
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(stock_data[:split])
    test_scaled  = scaler.transform(stock_data[split:])
    scaled_data  = np.vstack((train_scaled, test_scaled))

    X, y = prepare_sequences(scaled_data)
    n_train = split - 60
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # ── LSTM ─────────────────────────────────────────────────────────────────
    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.1),
        LSTM(32),
        Dropout(0.1),
        Dense(2)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32,
                   validation_data=(X_test, y_test),
                   callbacks=[es], verbose=0)

    # ── GRU ──────────────────────────────────────────────────────────────────
    gru_model = Sequential([
        GRU(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.1),
        GRU(32),
        Dropout(0.1),
        Dense(2)
    ])
    gru_model.compile(optimizer='adam', loss='mse')
    gru_model.fit(X_train, y_train, epochs=50, batch_size=32,
                  validation_data=(X_test, y_test),
                  callbacks=[es], verbose=0)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    lstm_pred_s = lstm_model.predict(X_test, verbose=0)
    gru_pred_s  = gru_model.predict(X_test, verbose=0)

    lstm_rmse = float(np.sqrt(mean_squared_error(y_test, lstm_pred_s)))
    gru_rmse  = float(np.sqrt(mean_squared_error(y_test, gru_pred_s)))

    best_name  = 'LSTM' if lstm_rmse <= gru_rmse else 'GRU'
    best_model = lstm_model if best_name == 'LSTM' else gru_model
    best_pred_s = lstm_pred_s if best_name == 'LSTM' else gru_pred_s

    best_real = inverse_pred(best_pred_s, scaler)

    dummy_y = np.zeros((len(y_test), 5))
    dummy_y[:, 0] = y_test[:, 0]
    dummy_y[:, 3] = y_test[:, 1]
    y_real = scaler.inverse_transform(dummy_y)

    # ── Next-day forecast ─────────────────────────────────────────────────────
    last_60 = scaled_data[-60:].reshape(1, 60, 5)
    nxt_s   = best_model.predict(last_60, verbose=0)
    nxt_r   = inverse_pred(nxt_s, scaler)

    current_close    = float(df['Close'].iloc[-1])
    predicted_close  = float(nxt_r[0, 3])
    predicted_open   = float(nxt_r[0, 0])
    change_pct       = (predicted_close - current_close) / current_close * 100
    direction        = 'UP' if predicted_close >= current_close else 'DOWN'

    # ── Chart data ───────────────────────────────────────────────────────────
    hist_dates = df.index.strftime('%Y-%m-%d').tolist()
    hist_close = [round(float(x), 2) for x in df['Close'].values.flatten()]

    test_df    = df.iloc[split:]
    test_dates = test_df.index[-len(y_real):].strftime('%Y-%m-%d').tolist()
    actual_close = [round(float(x), 2) for x in y_real[:, 3]]
    pred_close   = [round(float(x), 2) for x in best_real[:, 3]]

    # ── Fundamentals ─────────────────────────────────────────────────────────
    def fmt_pct(v):
        if v == "N/A": return "N/A"
        return f"{float(v)*100:.2f}%" if abs(float(v)) < 10 else f"{float(v):.2f}%"

    pe    = safe_get(info, 'trailingPE', 'forwardPE')
    pb    = safe_get(info, 'priceToBook')
    mkcap = safe_get(info, 'marketCap')
    eps   = safe_get(info, 'trailingEps')
    div   = safe_get(info, 'dividendYield')
    bv    = safe_get(info, 'bookValue')
    promo = safe_get(info, 'heldPercentInsiders')
    roe   = safe_get(info, 'returnOnEquity')
    roce  = safe_get(info, 'returnOnAssets')
    sector = safe_get(info, 'sector', 'industry', default='—')
    name   = safe_get(info, 'longName', 'shortName', default=symbol)

    # 52-week
    high52 = safe_get(info, 'fiftyTwoWeekHigh')
    low52  = safe_get(info, 'fiftyTwoWeekLow')
    day_high = safe_get(info, 'dayHigh')
    day_low  = safe_get(info, 'dayLow')
    prev_close = safe_get(info, 'previousClose')

    day_change    = "N/A"
    day_change_pct = "N/A"
    if prev_close != "N/A":
        dc = current_close - float(prev_close)
        day_change     = f"{'+' if dc >= 0 else ''}{dc:.2f}"
        day_change_pct = f"{'+' if dc >= 0 else ''}{dc/float(prev_close)*100:.2f}%"

    return {
        # Identity
        "symbol":    symbol,
        "name":      name,
        "sector":    sector,
        # Price
        "current_close":   round(current_close, 2),
        "day_change":      day_change,
        "day_change_pct":  day_change_pct,
        "day_high":        round(float(day_high), 2)  if day_high  != "N/A" else "N/A",
        "day_low":         round(float(day_low), 2)   if day_low   != "N/A" else "N/A",
        "week52_high":     round(float(high52), 2)    if high52    != "N/A" else "N/A",
        "week52_low":      round(float(low52), 2)     if low52     != "N/A" else "N/A",
        # Prediction
        "best_model":      best_name,
        "lstm_rmse":       round(lstm_rmse, 4),
        "gru_rmse":        round(gru_rmse, 4),
        "next_open":       round(predicted_open, 2),
        "next_close":      round(predicted_close, 2),
        "change_pct":      round(change_pct, 2),
        "direction":       direction,
        # Fundamentals
        "pe":    round(float(pe), 2)  if pe  != "N/A" else "N/A",
        "pb":    round(float(pb), 2)  if pb  != "N/A" else "N/A",
        "eps":   round(float(eps), 2) if eps != "N/A" else "N/A",
        "mkcap": format_large(mkcap),
        "div_yield": fmt_pct(div)  if div  != "N/A" else "N/A",
        "book_value": round(float(bv), 2) if bv != "N/A" else "N/A",
        "promo_hold": fmt_pct(promo) if promo != "N/A" else "N/A",
        "roe":  fmt_pct(roe)  if roe  != "N/A" else "N/A",
        "roce": fmt_pct(roce) if roce != "N/A" else "N/A",
        # Chart
        "hist_dates":    hist_dates,
        "hist_close":    hist_close,
        "test_dates":    test_dates,
        "actual_close":  actual_close,
        "pred_close":    pred_close,
    }


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    body   = request.get_json(silent=True) or {}
    raw    = body.get('symbol', '').strip()
    if not raw:
        return jsonify({'error': 'Please enter a stock symbol.'}), 400

    symbol = clean_symbol(raw)
    try:
        result = run_prediction(symbol)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
