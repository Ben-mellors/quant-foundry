# Volatility Targeting for ETF Allocation

An independent, self-directed research note and Python backtesting workflow
exploring volatility targeting as a systematic approach to ETF allocation.

## Overview

The study tests whether scaling portfolio exposure by recent volatility improves
risk-adjusted returns compared to a static buy-and-hold benchmark, using daily
ETF data across SPY and QQQ.

- **Universe:** SPY, QQQ
- **Method:** inverse-volatility weighting with portfolio-level volatility targeting
- **Metrics:** total return multiple, Sharpe ratio, maximum drawdown
- **Data:** daily adjusted prices via yfinance (educational use)

## Files

- `volatility-targeting-report.pdf` - research note (methods, formulas, results, limitations)
- `backtest.py` - backtesting code and output generation
- `equity_curve.png` - equity curve plot used in the note

## How to run

Install dependencies:

`pip install pandas numpy matplotlib yfinance`

Run the backtest:

`python backtest.py`
​

## Limitations

Results are in-sample over a single historical period, use simplified transaction
cost assumptions, and are not intended as investment advice. The work is a
self-taught research exercise rather than university coursework.
