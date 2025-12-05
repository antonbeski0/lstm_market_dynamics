# LSTM Market Dynamics -- Advanced Stock Prediction Pipeline

A fully automated, production-grade **LSTM-based stock market
forecasting system** that integrates:

-   **Yahoo Finance OHLCV data**
-   **Over 80+ technical indicators**
-   **News sentiment (VADER + NewsAPI)**
-   **Insider trading activity (Finnhub)**
-   **Monte Carlo Dropout for uncertainty estimation**
-   **Bidirectional LSTM + Conv1D hybrid deep learning model**
-   **Caching system for ultra-fast training**
-   **Auto-generated forecasts & summary reports**

------------------------------------------------------------------------

## 🚀 Features

### ✔ Multi‑Source Market Intelligence

-   OHLCV data from Yahoo Finance\
-   News sentiment with VADER\
-   Insider trades from Finnhub

### ✔ Enhanced Feature Engineering

-   5--200 day SMAs & EMAs\
-   Volatility, ATR, RSI, MACD, Bollinger Bands\
-   Momentum, ROC, Stochastic Oscillators\
-   Price action features & ADX trend strength

### ✔ Deep Learning Architecture

-   Conv1D feature extractor\
-   Bidirectional LSTM layers\
-   MC Dropout (Bayesian uncertainty)\
-   Precision, Recall, AUC tracking

### ✔ High‑Performance Pipeline

-   RobustScaler normalization\
-   Automatic dataset windowing\
-   Smart train/val/test splitting\
-   Saved scaler, model & processed files

------------------------------------------------------------------------

## 📂 Folder Structure

    /TrainingData/indicators_data/processed/stocksData   # Raw + enriched csv files
    /cache                                                # Preprocessed windows
    /models                                               # Saved keras model
    /forecasts                                            # Forecast CSV files
    training_history.png                                  # Training graph
    lstm_market_dynamics.py                               # Main pipeline

------------------------------------------------------------------------

## 🛠 Installation

``` bash
pip install -r requirements.txt
```

Required libraries: - tensorflow - numpy, pandas - yfinance -
vaderSentiment - finnhub-python - newsapi-python - scikit-learn -
matplotlib

------------------------------------------------------------------------

## ▶ Usage

Simply run:

``` bash
python lstm_market_dynamics.py
```

This will: 1. Fetch + cache stock data\
2. Compute all indicators\
3. Train the LSTM model\
4. Generate predictions\
5. Save forecast CSVs + summary report

------------------------------------------------------------------------

## 📊 Outputs

### Per‑stock forecast file:

    /forecasts/RELIANCE_NS_forecast.csv

Columns: - Prob_Up - Uncertainty - Predicted_Direction - Confidence -
Signal (BUY / SELL / HOLD)

### Global summary:

    /forecasts/_SUMMARY_REPORT.csv

------------------------------------------------------------------------

## 🧠 Model Architecture (Summary)

-   Conv1D → BatchNorm → MC Dropout\
-   BiLSTM (128) → BatchNorm → MC Dropout\
-   BiLSTM (64) → BatchNorm → MC Dropout\
-   Dense layers with dropout\
-   Sigmoid output for direction classification

------------------------------------------------------------------------

## ⚠ Notes

-   NewsAPI & Finnhub require valid API keys.\
-   Modify the `CONFIG` object in the script to customize tickers,
    directories, thresholds, etc.

------------------------------------------------------------------------

## 📌 Author

Developed by **Anton Beski** -- for high‑accuracy market signal
generation and research-grade modelling.

------------------------------------------------------------------------

## ⭐ Support

If you find this project useful, please leave a ⭐ on GitHub!
