# Volatility Targeting for ETF Allocation (Independent Student Research Note)

This repo contains an independent, self-directed research note and a small Python backtesting workflow exploring volatility targeting for ETF allocation.

## Contents
- `paper/` - PDF research note (methods, formulas, results, limitations)
- `src/` - Python code used to run the backtests and generate outputs
- `figures/` - plots used in the paper

## Summary
- Universe: SPY, QQQ (ETFs)
- Method: inverse-vol weighting + portfolio-level volatility targeting (no leverage in final comparison)
- Metrics: total return multiple, Sharpe ratio, max drawdown
- Data: daily prices via yfinance (educational use)

## How to run
```bash
pip install -r requirements.txt
python src/main.py

