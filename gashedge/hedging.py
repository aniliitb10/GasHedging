"""
Hedging mathematics and simulations.

Notation used throughout the tutorial (see Appendix C):
    S   : price of the exposure we hold (e.g. PEG Dec-26, or a season strip)
    F   : price of the hedge instrument(s) (e.g. TTF front month)
    Q_S : exposure size in MWh (positive = long gas)
    h   : hedge ratio, number of hedge MWh per exposure MWh (we go *short* h*Q_S)
    dX  : one-period change in X
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gashedge.logging_config import get_logger

log = get_logger(__name__)


# ------------------------------------------------------------------- OLS -----

@dataclass
class OLSResult:
    beta: np.ndarray          # coefficients (intercept first if fitted)
    se: np.ndarray            # standard errors
    resid: np.ndarray
    r2: float
    n: int
    names: list[str]

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({"coef": self.beta, "std_err": self.se, "t_stat": self.beta / self.se},
                            index=self.names)


def ols(y, X, intercept: bool = True, names: list[str] | None = None) -> OLSResult:
    """
    Ordinary least squares via the normal equations (Appendix A):

        beta_hat = (X'X)^{-1} X'y
        Var(beta_hat) = s^2 (X'X)^{-1},  s^2 = e'e / (n - k)
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if names is None:
        names = [f"x{i}" for i in range(X.shape[1])]
    if intercept:
        X = np.column_stack([np.ones(len(y)), X])
        names = ["const"] + list(names)
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    n, k = X.shape
    s2 = resid @ resid / (n - k)
    se = np.sqrt(np.diag(s2 * XtX_inv))
    tss = ((y - y.mean()) ** 2).sum() if intercept else (y ** 2).sum()
    r2 = 1 - (resid @ resid) / tss
    return OLSResult(beta=beta, se=se, resid=resid, r2=float(r2), n=n, names=list(names))


# ------------------------------------------------------- hedge ratios --------

def min_variance_hedge_ratio(dS, dF) -> float:
    """
    h* = Cov(dS, dF) / Var(dF) = rho * sigma_S / sigma_F     (Chapter 3.2)

    Minimises Var(dS - h dF).  Identical to the OLS slope of dS on dF.
    """
    dS, dF = np.asarray(dS, float), np.asarray(dF, float)
    return float(np.cov(dS, dF, ddof=1)[0, 1] / np.var(dF, ddof=1))


def hedge_effectiveness(dS, dF, h: float) -> float:
    """1 - Var(dS - h dF) / Var(dS).  Equals R^2 when h is the OLS slope."""
    dS, dF = np.asarray(dS, float), np.asarray(dF, float)
    return float(1 - np.var(dS - h * dF, ddof=1) / np.var(dS, ddof=1))


def multi_instrument_hedge(dS, dF: pd.DataFrame | np.ndarray) -> np.ndarray:
    """
    Vector of hedge ratios for several instruments (Chapter 3.3):

        h* = Sigma_FF^{-1} Sigma_FS

    i.e. the multiple-regression slopes of dS on the columns of dF.
    """
    dF = np.asarray(dF, float)
    dS = np.asarray(dS, float).ravel()
    cov = np.cov(np.column_stack([dF, dS]).T, ddof=1)
    k = dF.shape[1]
    return np.linalg.solve(cov[:k, :k], cov[:k, k])


def rolling_hedge_ratio(dS: pd.Series, dF: pd.Series, window: int) -> pd.Series:
    """Rolling-window minimum-variance hedge ratio (Chapter 3.5)."""
    cov = dS.rolling(window).cov(dF)
    var = dF.rolling(window).var()
    return (cov / var).rename(f"h_rolling_{window}")


def ewma_hedge_ratio(dS: pd.Series, dF: pd.Series, halflife: float) -> pd.Series:
    """Exponentially weighted hedge ratio - reacts faster to regime changes."""
    cov = dS.ewm(halflife=halflife).cov(dF)
    var = dF.ewm(halflife=halflife).var()
    return (cov / var).rename(f"h_ewma_{halflife}")


# ------------------------------------------------------ P&L accounting -------

def hedged_pnl(dS, dF, q_s: float, h: float, dF2=None, h2: float = 0.0) -> pd.DataFrame:
    """
    Period-by-period P&L of a long exposure q_s (MWh) hedged with a short of h*q_s in F
    (and optionally h2*q_s in a second instrument).

        PnL_t = q_s * dS_t - h q_s dF_t - h2 q_s dF2_t
    """
    dS = pd.Series(np.asarray(dS, float))
    dF = pd.Series(np.asarray(dF, float))
    exposure = q_s * dS
    hedge = -h * q_s * dF
    if dF2 is not None:
        hedge = hedge - h2 * q_s * pd.Series(np.asarray(dF2, float))
    out = pd.DataFrame({"exposure": exposure, "hedge": hedge})
    out["net"] = out["exposure"] + out["hedge"]
    return out


def pnl_stats(pnl: pd.Series, name: str = "") -> dict[str, float]:
    p = np.asarray(pnl, float)
    return {"name": name, "mean": p.mean(), "std": p.std(ddof=1), "min": p.min(),
            "q01": np.quantile(p, 0.01), "q99": np.quantile(p, 0.99), "total": p.sum()}


# ------------------------------------------------- transaction costs --------

def execution_cost(lots: float, bid_ask: float, lot_mwh: float, depth_lots: float,
                   impact_coef: float = 0.5) -> float:
    """
    EUR cost of crossing the spread and walking the book (square-root impact law):

        cost = |lots| * lot_mwh * ( bid_ask/2 + impact_coef * bid_ask * sqrt(|lots| / depth_lots) )

    Half-spread is paid on every lot; impact grows with size relative to depth.
    """
    lots = abs(lots)
    if lots == 0:
        return 0.0
    per_mwh = bid_ask / 2 + impact_coef * bid_ask * np.sqrt(lots / depth_lots)
    return float(lots * lot_mwh * per_mwh)


# ------------------------------------------------- stack and roll ------------

def stack_and_roll(curve_hist: pd.DataFrame, exposure_tenor: int, exposure_mwh: float,
                   roll_every: int = 21, hedge_col: str = "M1", ratio: float = 1.0) -> pd.DataFrame:
    """
    Hedge a far-dated exposure (constant-maturity column M{exposure_tenor}) by
    shorting `ratio` x exposure in the front contract and rolling it every `roll_every` days.

    Uses the constant-maturity history, so the "roll" is modelled as re-entering
    the front-month column after paying the M1-M2 roll spread at each roll date.
    Returns daily exposure P&L, hedge P&L, roll cost and margin (variation) flows.
    """
    s = curve_hist[f"M{exposure_tenor}"]
    f = curve_hist[hedge_col]
    f_next = curve_hist["M2"] if hedge_col == "M1" else curve_hist[hedge_col]
    dS = s.diff().fillna(0.0)
    dF = f.diff().fillna(0.0)
    df = pd.DataFrame(index=curve_hist.index)
    df["exposure_pnl"] = exposure_mwh * dS
    df["hedge_pnl"] = -ratio * exposure_mwh * dF
    roll_idx = np.arange(roll_every, len(df), roll_every)
    roll_cost = np.zeros(len(df))
    # Rolling a short: buy back M1, sell M2.  Cost = (M1 - M2) per MWh when in backwardation (M1 > M2).
    roll_cost[roll_idx] = -ratio * exposure_mwh * (f.iloc[roll_idx].values - f_next.iloc[roll_idx].values)
    df["roll_pnl"] = roll_cost
    df["net_pnl"] = df["exposure_pnl"] + df["hedge_pnl"] + df["roll_pnl"]
    df["cum_exposure"] = df["exposure_pnl"].cumsum()
    df["cum_hedge_cash"] = (df["hedge_pnl"] + df["roll_pnl"]).cumsum()   # futures: cash daily
    df["cum_net"] = df["net_pnl"].cumsum()
    log.info("Stack-and-roll M%d with %s: net std %.0f EUR/day, cum roll P&L %.0f EUR", exposure_tenor,
             hedge_col, df.net_pnl.std(), df.roll_pnl.sum())
    return df


# ----------------------------------------------------- band hedging ----------

def band_hedging(inventory: pd.Series, prices: pd.Series, lot_mwh: float, band_lots: float,
                 bid_ask: float, depth_lots: float, target: float = 0.0) -> pd.DataFrame:
    """
    Hedge only when |inventory + hedge| exceeds `band_lots`; then trade back to `target`.

    Returns daily net position (lots), hedge trades, transaction costs and P&L.
    A band of 0 is 'hedge every fill'; a band of infinity is 'never hedge'.
    """
    inv = inventory.values.astype(float)
    px = prices.reindex(inventory.index).ffill().values
    hedge_pos = 0.0
    hedge_trades = np.zeros(len(inv))
    costs = np.zeros(len(inv))
    net = np.zeros(len(inv))
    for t in range(len(inv)):
        net_pos = inv[t] + hedge_pos
        if abs(net_pos) > band_lots:
            trade = target - net_pos
            hedge_pos += trade
            hedge_trades[t] = trade
            costs[t] = execution_cost(trade, bid_ask, lot_mwh, depth_lots)
        net[t] = inv[t] + hedge_pos
    df = pd.DataFrame({"inventory": inv, "hedge_trade": hedge_trades, "net_position": net,
                       "cost": costs, "price": px}, index=inventory.index)
    dpx = np.diff(px, prepend=px[0])
    df["pnl"] = np.roll(df["net_position"].values, 1) * lot_mwh * dpx - df["cost"]
    df.loc[df.index[0], "pnl"] = -df["cost"].iloc[0]
    return df


def band_sweep(inventory, prices, lot_mwh, bands, bid_ask, depth_lots) -> pd.DataFrame:
    rows = []
    for b in bands:
        r = band_hedging(inventory, prices, lot_mwh, b, bid_ask, depth_lots)
        rows.append({"band_lots": b, "pnl_std": r.pnl.std(), "total_cost": r.cost.sum(),
                     "n_trades": int((r.hedge_trade != 0).sum()), "lots_traded": r.hedge_trade.abs().sum(),
                     "worst_day": r.pnl.min()})
    return pd.DataFrame(rows)


# ----------------------------------------------------- strip decomposition ---

def strip_to_monthly_exposure(strip_mwh_per_hour: float, months: list[tuple[int, int]]) -> pd.DataFrame:
    """Decompose a strip (MW) into MWh exposure per delivery month, using DST-aware hours."""
    from gashedge.contracts import contract_symbol, delivery_hours
    rows = [{"year": y, "month": m, "symbol": contract_symbol(y, m), "hours": delivery_hours(y, m),
             "mwh": strip_mwh_per_hour * delivery_hours(y, m)} for y, m in months]
    return pd.DataFrame(rows)


__all__ = [
    "OLSResult", "ols", "min_variance_hedge_ratio", "hedge_effectiveness", "multi_instrument_hedge",
    "rolling_hedge_ratio", "ewma_hedge_ratio", "hedged_pnl", "pnl_stats", "execution_cost",
    "stack_and_roll", "band_hedging", "band_sweep", "strip_to_monthly_exposure",
]
