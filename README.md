# Characteristic-Managed Momentum (CMM) Replication

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DorLev165/cmm-momentum-replication/blob/master/notebooks/demo.ipynb)

A Python replication of the Characteristic-Managed Momentum strategy. A feed-forward neural network learns optimal softmax weights over a stock's past 231 daily returns, conditioned on firm characteristics, to predict next-month cross-sectional returns and construct long/short portfolios.

## How It Works

1. **Data**: Downloads S&P 500 price histories via yfinance (fast mode) or pulls the full JKP panel of 153 firm characteristics from WRDS (full replication mode)
2. **Model**: 3-layer FFN (32-16-8, following Gu-Kelly-Xiu 2020 NN3) with softmax output weighting daily returns. Ensemble of 5 seeds averaged to reduce variance
3. **Portfolio**: Mean-variance optimization with Ledoit-Wolf covariance shrinkage. Expanding training window (train 1973-1982, test 1983-1984, expand by 2 years, repeat through 2026)
4. **Analysis**: HML (High-Minus-Low) decile portfolio returns, regime-conditional breakdowns (momentum crash vs. non-crash), volatility-managed overlay (Daniel-Moskowitz 2016)

## Features

- Two data modes: yfinance (quick, survivorship-biased) or JKP/WRDS (153 characteristics, full replication)
- Expanding window with per-window result persistence
- Regime-conditional performance analysis
- Volatility-managed HML overlay (Barroso & Santa-Clara 2015)
- Industry-adjusted returns (Lewellen 2015, optional)
- Standalone portfolio simulation and rolling volatility utilities

## Project Structure

```
cmm/                  # Core CMM package
  model.py            # CMMModel (data loading, feature prep)
  ffn.py              # PyTorch FFN architecture
  training.py         # Expanding window training loop
  portfolio.py        # HML decile portfolio construction
  regime.py           # Regime-conditional analysis
  covariance.py       # Ledoit-Wolf shrinkage
  fetch_data.py       # yfinance data fetcher
  fetch_data_jkp.py   # WRDS/JKP data fetcher
main.py               # Primary entry point
simulate_portfolio.py  # Portfolio NAV simulation
rolling_volatility.py  # Rolling volatility analysis
scripts/              # Utility scripts
tests/                # Test suite
```

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

For WRDS mode, set `WRDS_USERNAME` in a `.env` file.

## Tech Stack

PyTorch, NumPy, pandas, scikit-learn, yfinance, matplotlib, WRDS (optional)
