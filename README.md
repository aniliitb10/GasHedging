# Hedging European Gas Futures — a hands-on tutorial

A notebook-based tutorial on hedging accumulated gas risk on ICE (TTF benchmark, PEG basis), written for a
senior engineer moving into the hedging side of an electronic-trading business.

Start here: **`tutorial/00_introduction.ipynb`**.

## Layout

```
tutorial/
  00_introduction.ipynb                 storyline, venue notes, map of all chapters
  module1_market_structure/  1.1–1.5    contracts, curve & strips, liquidity, curve dynamics, hubs & basis
  module2_hedging_core/      2.1–2.6    why hedge, instrument menu, 1:1 mismatches, stack-and-roll,
                                         bands & costs, the hedging pipeline
  module3_risk_analysis/     3.1–3.5    linear model, min-variance ratio (OLS), multi-instrument & PCA,
                                         VaR/ES/stress, estimation error & regimes
  appendix/                  A–C        least squares derived; delivery hours/DST/strips; notation, PCA, references
gashedge/                                shared library used by every notebook
  contracts.py      month codes, DST-aware delivery hours, expiries, strips
  market_data.py    synthetic (clearly labelled) curves, price histories, liquidity estimates
  hedging.py        OLS, hedge ratios, execution cost, stack-and-roll, band hedging
  risk.py           linear P&L model, VaR/ES, Kupiec backtest, PCA, stress
  plotting.py       seaborn helpers
  logging_config.py millisecond-precision logging
tools/
  build_all.py      regenerates and executes all notebooks from tools/content/*.py
```

Every chapter opens with the problem it solves, derives the solution with equations and code, and ends with
the pitfall that motivates the next chapter plus a set of questions whose answers are hidden until clicked.

## Running

Open any notebook with the project's `.venv` kernel (pandas, numpy, scipy, matplotlib, seaborn, jupyter) and
run top to bottom. Notebooks import `gashedge` by locating the project root automatically.

To regenerate the notebooks after editing the content sources:

```
.venv/bin/python tools/build_all.py --execute
```

## Data disclaimer

No real market data is used. Curves, histories and liquidity figures are **stylised, seeded simulations**
designed to reproduce documented qualitative features of European gas markets (seasonality, Samuelson
volatility decay, hub basis mean reversion, 2022-style decoupling). Liquidity numbers live in one table
(`gashedge.market_data.LIQUIDITY_BUCKETS`) meant to be replaced with exchange statistics. Verify contract
specifications against the current ICE product documentation before production use.
