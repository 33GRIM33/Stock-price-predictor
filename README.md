# StockSight

StockSight is a Flask-based stock prediction dashboard for Indian stocks (NSE/Nifty-style symbols). It combines historical price analysis with deep learning models and key financial fundamentals to generate a simple forecast and summary for a stock.

## Overview

The app allows a user to enter a stock symbol such as `RELIANCE`, `TCS`, or `INFY`. It then:

- fetches historical market data from Yahoo Finance
- preprocesses the data for sequence modeling
- trains and compares LSTM and GRU models
- estimates the next-day price direction and movement
- displays a chart of historical vs. predicted close prices
- shows relevant valuation and performance metrics such as P/E, P/B, market cap, ROE, and dividend yield

This project is intended for learning, experimentation, and demonstration purposes, not as a guaranteed financial recommendation tool.

## Features

- Stock symbol search and prediction workflow
- Automatic `.NS` suffix handling for Indian stocks
- LSTM and GRU model comparison
- Next-day open/close forecast
- Historical price chart and validation chart
- Fundamental metrics panel
- Responsive web interface built with Flask and HTML/CSS/JavaScript

## Tech Stack

- Python
- Flask
- Yahoo Finance (`yfinance`)
- Pandas and NumPy
- Scikit-learn
- TensorFlow / Keras
- Chart.js

## Project Structure

```text
stp/
├── app/
│   ├── app.py
│   ├── model_final.ipynb
│   └── templates/
│       └── index.html
├── requirements.txt
├── README.md
└── .venv/
```

## Installation

1. Open a terminal in the project root.
2. Create and activate a virtual environment:

   Windows PowerShell:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   macOS/Linux:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the project dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run the App

From the project root, start the Flask server:

```bash
python app/app.py
```

Then open the app in your browser:

```text
http://127.0.0.1:5000/
```

## Usage

1. Enter a stock symbol in the search box.
2. Click the prediction button.
3. View the forecast, chart, and key fundamentals for the selected stock.

The app automatically appends `.NS` for Indian stocks if it is not already included.

## Notes

- Market data is fetched live from Yahoo Finance, so availability depends on internet access and Yahoo service status.
- Deep learning predictions are based on historical patterns and should not be treated as financial advice.
- Model training can take some time depending on your system and the selected stock.

## License

This project is for educational and demonstration use.
