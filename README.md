# BTC-USD-Deep-Learning-Trading-Signal-Engine


A Deep Learning-based quantitative trading framework for Bitcoin directional forecasting using Bidirectional LSTM networks, financial feature engineering, dynamic risk management, and forward-validation backtesting.

## Overview

This project predicts the next-day direction of Bitcoin (BTC-USD) using a Bidirectional LSTM neural network trained on engineered market, volatility, volume, and technical-analysis features.

The framework includes:

* Signal generation (LONG / SHORT / CASH)
* Dynamic volatility-aware thresholds
* ATR-based Take Profit / Stop Loss
* Backtesting engine
* Performance evaluation metrics
* Strategy vs Buy-and-Hold comparison

---

## Features

### Market Features

* Multi-period returns
* Momentum acceleration
* Mean reversion indicators
* SMA crossovers
* Distance from 52-week highs/lows

### Volatility Features

* ATR
* Parkinson Volatility
* Rolling volatility regimes

### Volume Features

* Volume spikes
* Volume trend analysis
* Volume-price divergence

### Candlestick Features

* Candle body ratio
* Upper/lower wick structure
* Gap analysis

### Additional Features

* Bitcoin/Gold relative strength
* Day-of-week cyclic encoding

---

## Model Architecture

Input → Bidirectional LSTM (64) → BatchNorm → Dropout → Bidirectional LSTM (32) → Dense → Sigmoid Output

Framework:

* TensorFlow / Keras
* Scikit-Learn
* Pandas / NumPy

---

## Trading Logic

The model outputs bullish/bearish probabilities.

Trading signals are generated dynamically using:

* volatility-adjusted thresholds
* trend filters
* adaptive confidence zones

Signals:

* LONG
* SHORT
* CASH

---

## Risk Management

The strategy includes:

* ATR-based Stop Loss
* ATR-based Take Profit
* Dynamic position sizing
* Volatility-aware exits

Risk-to-Reward Ratio:
1 : 2

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Sharpe Ratio
* Alpha vs Buy & Hold

---

## Future Improvements

Potential upgrades:

* Transformer architectures
* Attention mechanisms
* Sentiment analysis
* Reinforcement learning
* Order-book data
* Hyperparameter optimization
* Multi-asset trading

---

## Technologies

* Python
* TensorFlow / Keras
* Scikit-Learn
* Pandas
* NumPy
* Matplotlib
* yFinance

---

## Disclaimer

This project is for research and educational purposes only and should not be considered financial advice.
