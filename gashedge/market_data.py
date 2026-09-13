"""
Synthetic - but structurally realistic - market data for the tutorial.

IMPORTANT: nothing here is real market data.  Levels, volumes and spreads are
*stylised estimates* chosen to reproduce well-documented qualitative features of
European gas markets (winter premium, liquidity concentrated at the front,
Samuelson volatility decay, TTF/PEG basis mean reversion, 2021-22 style crisis).
Every generator takes a `seed` so results are reproducible, and every
liquidity number lives in one table (`LIQUIDITY_BUCKETS`) that you should
replace with your own exchange statistics when you have them.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from gashedge.contracts import (
    TTF_SPEC, ContractSpec, calendar_months, contract_table, delivery_hours,
    quarter_months, season_months, strip_label, strip_price,
)
from gashedge.logging_config import get_logger

log = get_logger(__name__)

# ------------------------------------------------------------ seasonality ----

# Multiplicative seasonal shape of the forward curve relative to the annual mean.
# Winter demand (heating) puts a premium on Dec-Feb; summer is injection season.
SEASONAL_FACTOR = {1: 1.12, 2: 1.10, 3: 1.03, 4: 0.94, 5: 0.92, 6: 0.92,
                   7: 0.93, 8: 0.94, 9: 0.96, 10: 1.02, 11: 1.07, 12: 1.13}


def seasonal_factor(month: int) -> float:
    return SEASONAL_FACTOR[month]


# --------------------------------------------------------------- liquidity ----

# Stylised liquidity by tenor bucket for a TTF-like benchmark hub.  Columns:
#   rel_adv     : average daily volume relative to the front month (front = 1.0)
#   bid_ask     : typical top-of-book spread in EUR/MWh (calm market)
#   depth_lots  : typical size available at the touch (lots)
# PEG-like hubs are scaled by `HUB_LIQUIDITY_SCALE`.
LIQUIDITY_BUCKETS = pd.DataFrame(
    [
        ("M1", 1.00, 0.010, 40),
        ("M2", 0.65, 0.015, 30),
        ("M3", 0.35, 0.020, 20),
        ("M4-M6", 0.12, 0.035, 10),
        ("M7-M12", 0.04, 0.060, 5),
        ("M13+", 0.01, 0.120, 2),
        ("Q1-Q2 (nearest quarters)", 0.30, 0.025, 15),
        ("Q3+ (further quarters)", 0.08, 0.050, 6),
        ("Season (nearest two)", 0.45, 0.020, 20),
        ("Season (further)", 0.10, 0.045, 8),
        ("Cal (nearest)", 0.35, 0.030, 12),
        ("Cal (further)", 0.05, 0.080, 4),
    ],
    columns=["bucket", "rel_adv", "bid_ask", "depth_lots"],
).set_index("bucket")

# Rough relative size of hubs versus TTF (TTF is the European benchmark by a wide margin).
HUB_LIQUIDITY_SCALE = {"TTF": 1.0, "NBP": 0.30, "PEG": 0.06, "THE": 0.08}

FRONT_MONTH_ADV_LOTS = 25_000  # stylised TTF front-month lots/day (outright + spread legs)


def month_bucket(tenor: int) -> str:
    if tenor == 1:
        return "M1"
    if tenor == 2:
        return "M2"
    if tenor == 3:
        return "M3"
    if tenor <= 6:
        return "M4-M6"
    if tenor <= 12:
        return "M7-M12"
    return "M13+"


def liquidity_for(bucket: str, hub: str = "TTF") -> dict[str, float]:
    row = LIQUIDITY_BUCKETS.loc[bucket]
    scale = HUB_LIQUIDITY_SCALE[hub]
    return {
        "bucket": bucket,
        "est_adv_lots": float(round(FRONT_MONTH_ADV_LOTS * row.rel_adv * scale)),
        "bid_ask": float(row.bid_ask / np.sqrt(scale)),      # thinner hubs -> wider spreads
        "depth_lots": float(max(1, round(row.depth_lots * scale))),
    }


# ----------------------------------------------------------- forward curve ----

def synthetic_forward_curve(
    as_of: date,
    n_months: int = 36,
    level: float = 32.0,
    annual_drift: float = -0.04,
    noise_bp: float = 0.004,
    spec: ContractSpec = TTF_SPEC,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Monthly forward curve = level * seasonal(month) * (1 + drift)^years_ahead * noise.

    `annual_drift` < 0 gives a gently backwardated curve (the post-crisis norm);
    set it > 0 for contango.  Returns the contract table enriched with prices
    and liquidity estimates.
    """
    rng = np.random.default_rng(seed)
    df = contract_table(as_of, n_months, spec)
    years_ahead = (df["tenor"] - 1) / 12.0
    seasonal = df["month"].map(SEASONAL_FACTOR)
    noise = np.exp(rng.normal(0, noise_bp, len(df)))
    df["price"] = (level * seasonal * (1 + annual_drift) ** years_ahead * noise).round(3)
    liq = pd.DataFrame([liquidity_for(month_bucket(t), spec.hub) for t in df["tenor"]])
    df = pd.concat([df, liq], axis=1)
    df["bid"] = (df["price"] - df["bid_ask"] / 2).round(3)
    df["ask"] = (df["price"] + df["bid_ask"] / 2).round(3)
    df["hub"] = spec.hub
    log.info("Synthetic %s curve as of %s: %d months, front=%.2f, level=%.1f", spec.hub, as_of,
             n_months, df["price"].iloc[0], level)
    return df


def monthly_price_dict(curve: pd.DataFrame) -> dict[tuple[int, int], float]:
    return {(int(r.year), int(r.month)): float(r.price) for r in curve.itertuples()}


def strip_table(curve: pd.DataFrame, hub: str = "TTF") -> pd.DataFrame:
    """Quarters, seasons and calendar years that are fully covered by the monthly curve."""
    prices = monthly_price_dict(curve)
    covered = set(prices)
    rows = []

    def add(kind, year, idx, months, order):
        if all(m in covered for m in months):
            hours = sum(delivery_hours(y, m) for y, m in months)
            rows.append({"strip": strip_label(kind, year, idx), "kind": kind, "start": date(months[0][0], months[0][1], 1),
                         "n_months": len(months), "hours": hours, "price": round(strip_price(prices, months), 3),
                         "months": months, "order_": order})

    years = sorted({y for y, _ in covered})
    q_order = s_order = c_order = 0
    for y in years:
        for q in range(1, 5):
            months = quarter_months(y, q)
            if all(m in covered for m in months):
                q_order += 1
                add("Q", y, q, months, q_order)
        for season in ("SUM", "WIN"):
            months = season_months(y, season)
            if all(m in covered for m in months):
                s_order += 1
                add(season, y, None, months, s_order)
        months = calendar_months(y)
        if all(m in covered for m in months):
            c_order += 1
            add("CAL", y, None, months, c_order)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    def bucket(row):
        if row.kind == "Q":
            return "Q1-Q2 (nearest quarters)" if row.order_ <= 2 else "Q3+ (further quarters)"
        if row.kind in ("SUM", "WIN"):
            return "Season (nearest two)" if row.order_ <= 2 else "Season (further)"
        return "Cal (nearest)" if row.order_ == 1 else "Cal (further)"

    liq = pd.DataFrame([liquidity_for(bucket(r), hub) for r in df.itertuples()])
    df = pd.concat([df.drop(columns="order_"), liq], axis=1).sort_values("start").reset_index(drop=True)
    return df


# ----------------------------------------------------------- price history ----

def simulate_hub_history(
    n_days: int = 1000,
    start: date = date(2023, 1, 2),
    ttf0: float = 35.0,
    daily_vol: float = 0.035,
    basis_mean: float = 0.6,
    basis_kappa: float = 0.08,
    basis_vol: float = 0.35,
    mean_reversion: float = 0.01,
    crisis: bool = False,
    seed: int = 11,
) -> pd.DataFrame:
    """
    Daily front-month prices for TTF, PEG and NBP (all expressed in EUR/MWh).

    TTF  : log price with GARCH-like volatility clustering (gas vol comes in bursts).
    PEG  : TTF + basis, basis is Ornstein-Uhlenbeck (mean reverting) around `basis_mean`.
           PEG usually trades at a small premium to TTF (transport / entry-exit costs).
    NBP  : TTF + a noisier basis (interconnector flows, GBP effects folded in).
    crisis=True inserts a 2021-22 style spike in the middle third: vol x2.5, level x4, PEG beta -> 0.7,
    and a basis that decouples (French LNG regas glut vs. continental scarcity).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)

    # --- TTF: GARCH(1,1)-flavoured log returns
    omega, alpha, beta = daily_vol**2 * 0.05, 0.10, 0.85
    sig2 = np.full(n_days, daily_vol**2)
    z = np.zeros(n_days)                 # GARCH innovations (unscaled)
    r = np.zeros(n_days)                 # realised log returns
    logp = np.full(n_days, np.log(ttf0))
    lo, hi = n_days // 3, 2 * n_days // 3
    for t in range(1, n_days):
        sig2[t] = omega + alpha * z[t - 1] ** 2 + beta * sig2[t - 1]
        z[t] = np.sqrt(sig2[t]) * rng.standard_t(df=5) / np.sqrt(5 / 3)
        if crisis and lo <= t < hi:
            r[t] = 2.5 * z[t]
            if t < lo + 60:
                r[t] += np.log(4.0) / 60          # ramp up x4 over ~3 months
            elif t >= hi - 90:
                r[t] -= np.log(3.0) / 90          # partial unwind
        else:
            # mild mean reversion in log price keeps calm-regime levels realistic
            r[t] = z[t] + mean_reversion * (np.log(ttf0) - logp[t - 1])
        logp[t] = logp[t - 1] + r[t]
    ttf = np.exp(logp)

    # --- PEG basis: OU in price space, vol scales with level (basis widens in expensive markets)
    # In a crisis the pipes are full: arbitrage (mean reversion) switches off and the basis
    # becomes a volatile process of its own, drifting to a large PEG discount.
    basis = np.zeros(n_days)
    basis[0] = basis_mean
    for t in range(1, n_days):
        lvl = ttf[t] / ttf0
        in_crisis = crisis and lo <= t < hi
        kappa = basis_kappa * (0.15 if in_crisis else 1.0)
        shock = basis_vol * (lvl if in_crisis else np.sqrt(lvl)) * rng.standard_t(4) / np.sqrt(2)   # fat-tailed basis jumps
        basis[t] = basis[t - 1] + kappa * (basis_mean - basis[t - 1]) + shock
    if crisis:
        basis[lo:hi] -= np.linspace(0, 8, hi - lo) * np.sin(np.linspace(0, np.pi, hi - lo))  # PEG discount blow-out
    # PEG's sensitivity to TTF moves (the true hedge ratio) is ~1 normally; in a crisis the French system,
    # long LNG regas capacity, under-reacts to continental scarcity: beta falls toward ~0.7.
    beta = np.ones(n_days)
    if crisis:
        ramp = np.clip((np.arange(n_days) - lo) / 40, 0, 1) * np.clip((hi - np.arange(n_days)) / 40, 0, 1)
        beta = 1.0 - 0.3 * ramp
    peg = np.empty(n_days)
    peg[0] = ttf[0] + basis[0]
    for t in range(1, n_days):
        peg[t] = peg[t - 1] + beta[t] * (ttf[t] - ttf[t - 1]) + (basis[t] - basis[t - 1])

    # --- NBP basis: noisier, weaker mean reversion
    nbasis = np.zeros(n_days)
    for t in range(1, n_days):
        nbasis[t] = nbasis[t - 1] + 0.04 * (-0.8 - nbasis[t - 1]) + 0.9 * rng.standard_normal()
    nbp = ttf + nbasis

    df = pd.DataFrame({"TTF": ttf, "PEG": peg, "NBP": nbp}, index=idx).round(3)
    df.index.name = "date"
    log.info("Simulated %d days of hub history (crisis=%s); TTF range %.1f-%.1f", n_days, crisis,
             df.TTF.min(), df.TTF.max())
    return df


def simulate_curve_history(
    n_days: int = 750,
    n_tenors: int = 24,
    start: date = date(2023, 1, 2),
    level0: float = 32.0,
    vol_front: float = 0.035,
    vol_back: float = 0.014,
    decay: float = 8.0,
    seed: int = 21,
) -> pd.DataFrame:
    """
    Daily history of a *constant-maturity* curve M1..Mn (log prices driven by 3 factors).

    dlogP_i = beta_i^level * L + beta_i^slope * S + beta_i^curv * C + idio_i

    with volatility of each tenor following the Samuelson effect:
        sigma_i = vol_back + (vol_front - vol_back) * exp(-(i-1)/decay)

    Constant-maturity means column M1 is always "the front month", which is
    the standard representation for factor / PCA analysis (Appendix C).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    tenors = np.arange(1, n_tenors + 1)
    sigma = vol_back + (vol_front - vol_back) * np.exp(-(tenors - 1) / decay)

    # factor loadings (normalised): level ~ flat, slope ~ linear, curvature ~ quadratic
    x = (tenors - tenors.mean()) / tenors.std()
    b_level = np.ones(n_tenors)
    b_slope = -x
    b_curv = (x**2 - (x**2).mean())
    b_curv /= np.abs(b_curv).max()
    factor_vol = np.array([0.80, 0.30, 0.15])   # share of variance explained ~ 70 / 20 / 5 %

    # Student-t shocks (df=4, unit variance): gas returns are fat-tailed, and so should the simulator be
    t_scale = 1 / np.sqrt(4 / (4 - 2))
    L, S, C = rng.standard_t(4, (3, n_days)) * t_scale * factor_vol[:, None]
    common = np.outer(L, b_level) + np.outer(S, b_slope) + np.outer(C, b_curv)
    common /= common.std(axis=0, keepdims=True)
    idio = rng.standard_t(4, (n_days, n_tenors)) * t_scale * 0.25
    dlog = (common + idio) / np.sqrt(1 + 0.25**2) * sigma
    # volatility clustering common to the whole curve (GARCH(1,1)-style multiplier with unit mean square)
    m2 = np.ones(n_days)
    for t in range(1, n_days):
        m2[t] = 0.05 + 0.12 * (dlog[t - 1, 0] / sigma[0]) ** 2 * m2[t - 1] / max(m2[t - 1], 1e-9) + 0.83 * m2[t - 1]
    dlog = dlog * np.sqrt(m2)[:, None]

    # NB: no seasonal shape here on purpose.  In a constant-maturity panel a fixed seasonal factor per
    # column would make "M1 - M2" a permanent spread (an artifact); we keep a gentle 3%/yr backwardation only.
    logp0 = np.log(level0 * 0.97 ** ((tenors - 1) / 12))
    logp = logp0 + np.cumsum(dlog, axis=0)
    df = pd.DataFrame(np.exp(logp), index=idx, columns=[f"M{i}" for i in tenors]).round(3)
    df.index.name = "date"
    log.info("Simulated constant-maturity curve history: %d days x %d tenors", n_days, n_tenors)
    return df


def simulate_mm_inventory(
    n_days: int = 250,
    fills_per_day: float = 40,
    lot_std: float = 2.0,
    skew: float = 0.01,
    quote_skew_reversion: float = 0.03,
    seed: int = 5,
) -> pd.Series:
    """
    Net inventory (lots) accumulated by a market-making book, day by day.

    Each fill is +/- N(0, lot_std) lots with a mild persistent skew (customers
    are net sellers/buyers for weeks at a time), which is why MM inventories
    drift rather than staying flat; quote skewing pulls them back slowly.  Positive = long gas.
    """
    rng = np.random.default_rng(seed)
    n_fills = rng.poisson(fills_per_day, n_days)
    drift = np.zeros(n_days)
    for t in range(1, n_days):
        drift[t] = 0.90 * drift[t - 1] + skew * rng.standard_normal()
    daily = np.array([rng.normal(drift[t], lot_std, n).sum() for t, n in enumerate(n_fills)])
    # market-makers skew quotes to attract offsetting flow, which pulls inventory back toward zero
    level = np.zeros(n_days)
    for t in range(n_days):
        level[t] = (level[t - 1] if t else 0.0) * (1 - quote_skew_reversion) + daily[t]
    inv = pd.Series(np.round(level), index=pd.bdate_range(date(2025, 1, 2), periods=n_days), name="inventory_lots")
    log.info("Simulated MM inventory: final %d lots, max |inv| %d lots", inv.iloc[-1], inv.abs().max())
    return inv


def business_days(start: date, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


__all__ = [
    "SEASONAL_FACTOR", "LIQUIDITY_BUCKETS", "HUB_LIQUIDITY_SCALE", "FRONT_MONTH_ADV_LOTS",
    "seasonal_factor", "month_bucket", "liquidity_for", "synthetic_forward_curve",
    "monthly_price_dict", "strip_table", "simulate_hub_history", "simulate_curve_history",
    "simulate_mm_inventory", "business_days", "timedelta",
]
