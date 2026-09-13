"""
gashedge - helper library for the European gas hedging tutorial.

Sub-modules
-----------
logging_config : millisecond-precision logger factory
contracts      : month codes, delivery hours, strips, expiries
market_data    : synthetic (clearly labelled) curves, histories and liquidity estimates
hedging        : hedge ratios, OLS, stack-and-roll, band hedging simulations
risk           : linear P&L model, VaR / ES, PCA, hedge effectiveness
plotting       : seaborn helpers shared by the notebooks
"""

from gashedge.logging_config import get_logger

__all__ = ["get_logger"]
__version__ = "0.1.0"
