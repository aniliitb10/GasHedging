"""
Risk measurement for a linear (delta) book.

The workhorse is the *linear model* (Chapter 3.1):

    dV  ~=  sum_i  Delta_i * dP_i  =  Delta' dP

where Delta_i is the MWh exposure to instrument i and dP_i its price change in
EUR/MWh.  Everything else - variance, VaR, hedge ratios, PCA - follows from
Cov(dP) once the book is written as a vector Delta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from gashedge.logging_config import get_logger

log = get_logger(__name__)


# ------------------------------------------------------- linear model -------

def linear_pnl(deltas: np.ndarray | pd.Series, dP: np.ndarray | pd.DataFrame) -> np.ndarray:
    """P&L_t = Delta' dP_t for every row t of dP."""
    return np.asarray(dP, float) @ np.asarray(deltas, float)


def portfolio_variance(deltas, cov) -> float:
    """Var(Delta' dP) = Delta' Sigma Delta."""
    d = np.asarray(deltas, float)
    return float(d @ np.asarray(cov, float) @ d)


def risk_contributions(deltas, cov) -> pd.Series:
    """
    Euler decomposition of portfolio volatility:
        sigma_p = sum_i Delta_i * (Sigma Delta)_i / sigma_p
    Each term is the marginal contribution of instrument i (sums to sigma_p).
    """
    d = np.asarray(deltas, float)
    cov = np.asarray(cov, float)
    sigma_p = np.sqrt(d @ cov @ d)
    contrib = d * (cov @ d) / sigma_p
    names = deltas.index if isinstance(deltas, pd.Series) else range(len(d))
    return pd.Series(contrib, index=names, name="vol_contribution")


# ------------------------------------------------------------- VaR / ES -----

def parametric_var(deltas, cov, alpha: float = 0.99, horizon_days: int = 1) -> float:
    """Gaussian VaR: z_alpha * sigma_p * sqrt(h).  Positive number = potential loss."""
    sigma = np.sqrt(portfolio_variance(deltas, cov) * horizon_days)
    return float(stats.norm.ppf(alpha) * sigma)


def parametric_es(deltas, cov, alpha: float = 0.99, horizon_days: int = 1) -> float:
    """Gaussian expected shortfall: sigma * phi(z_alpha) / (1 - alpha)."""
    sigma = np.sqrt(portfolio_variance(deltas, cov) * horizon_days)
    z = stats.norm.ppf(alpha)
    return float(sigma * stats.norm.pdf(z) / (1 - alpha))


def historical_var(pnl, alpha: float = 0.99) -> float:
    """Empirical quantile of losses (positive number)."""
    return float(-np.quantile(np.asarray(pnl, float), 1 - alpha))


def historical_es(pnl, alpha: float = 0.99) -> float:
    p = np.asarray(pnl, float)
    var = historical_var(p, alpha)
    tail = p[p <= -var]
    return float(-tail.mean()) if len(tail) else var


def var_backtest(pnl, var_series, alpha: float = 0.99) -> dict[str, float]:
    """Kupiec proportion-of-failures test: are exceedances consistent with 1-alpha?"""
    pnl = np.asarray(pnl, float)
    var_series = np.asarray(var_series, float)
    mask = ~np.isnan(var_series)
    exceed = (pnl[mask] < -var_series[mask])
    n, x = int(mask.sum()), int(exceed.sum())
    p = 1 - alpha
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    else:
        lr = -2 * (np.log((1 - p) ** (n - x) * p ** x) - np.log((1 - x / n) ** (n - x) * (x / n) ** x))
    return {"n": n, "exceedances": x, "expected": n * p, "kupiec_LR": float(lr),
            "p_value": float(1 - stats.chi2.cdf(lr, 1))}


# ------------------------------------------------------------------ PCA -----

def pca(returns: pd.DataFrame, n_components: int = 3) -> dict:
    """
    Principal components of curve moves (Appendix C).

    Sigma = V Lambda V'   ->   factor loadings V[:, :k], explained variance Lambda_k / sum(Lambda).
    Returns loadings (DataFrame), explained ratio, and factor scores.
    """
    X = returns.values - returns.values.mean(axis=0)
    cov = np.cov(X.T, ddof=1)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    # sign convention: first component positive (a 'level' shock raises all prices)
    for j in range(eigvec.shape[1]):
        if eigvec[:, j].sum() < 0:
            eigvec[:, j] *= -1
    loadings = pd.DataFrame(eigvec[:, :n_components], index=returns.columns,
                            columns=[f"PC{i + 1}" for i in range(n_components)])
    scores = pd.DataFrame(X @ eigvec[:, :n_components], index=returns.index, columns=loadings.columns)
    explained = eigval[:n_components] / eigval.sum()
    log.info("PCA explained variance: %s", np.round(explained, 3))
    return {"loadings": loadings, "explained": explained, "scores": scores, "eigval": eigval}


# --------------------------------------------------- stress scenarios -------

def stress_pnl(deltas: pd.Series, scenarios: dict[str, pd.Series | np.ndarray]) -> pd.Series:
    """Apply deterministic price shocks (EUR/MWh per instrument) to the delta vector."""
    out = {name: float(np.asarray(shock, float) @ deltas.values) for name, shock in scenarios.items()}
    return pd.Series(out, name="stress_pnl")


def bucket_deltas(monthly_deltas: pd.DataFrame, bucket_col: str = "bucket") -> pd.Series:
    """Aggregate a per-month delta ladder into tenor buckets (M1, M2, M3, M4-M6, ...)."""
    return monthly_deltas.groupby(bucket_col, sort=False)["mwh"].sum()


__all__ = [
    "linear_pnl", "portfolio_variance", "risk_contributions", "parametric_var", "parametric_es",
    "historical_var", "historical_es", "var_backtest", "pca", "stress_pnl", "bucket_deltas",
]
