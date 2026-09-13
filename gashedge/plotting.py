"""Seaborn-based plotting helpers shared by the notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from gashedge.logging_config import get_logger

log = get_logger(__name__)

PALETTE = "deep"


def setup_style() -> None:
    sns.set_theme(style="whitegrid", palette=PALETTE, context="notebook")
    plt.rcParams["figure.figsize"] = (11, 4.5)
    plt.rcParams["axes.titleweight"] = "bold"


def plot_curve(curve: pd.DataFrame, title: str = "Forward curve", ax=None, hue: str | None = None):
    ax = ax or plt.gca()
    sns.lineplot(data=curve, x="delivery_start", y="price", marker="o", ax=ax, hue=hue)
    ax.set_ylabel("EUR/MWh")
    ax.set_xlabel("Delivery month")
    ax.set_title(title)
    return ax


def plot_curve_with_strips(curve: pd.DataFrame, strips: pd.DataFrame, kinds=("Q", "SUM", "WIN", "CAL"), ax=None):
    ax = ax or plt.gca()
    sns.lineplot(data=curve, x="delivery_start", y="price", marker="o", ax=ax, label="Monthly", color="0.3")
    colors = dict(zip(kinds, sns.color_palette(PALETTE, len(kinds))))
    for r in strips[strips.kind.isin(kinds)].itertuples():
        start = pd.Timestamp(r.start)
        end = start + pd.DateOffset(months=r.n_months)
        ax.hlines(r.price, start, end, colors=colors[r.kind], lw=3, alpha=0.8)
        ax.text(start, r.price, r.strip, fontsize=8, va="bottom", color=colors[r.kind])
    ax.set_ylabel("EUR/MWh")
    ax.set_xlabel("Delivery period")
    ax.set_title("Monthly curve with fair strip prices (hours-weighted averages)")
    return ax


def plot_liquidity(curve: pd.DataFrame, ax=None):
    ax = ax or plt.gca()
    sns.barplot(data=curve, x="tenor", y="est_adv_lots", hue="bucket", dodge=False, ax=ax)
    ax.set_yscale("log")
    ax.set_ylabel("Estimated ADV (lots, log scale)")
    ax.set_xlabel("Tenor (months ahead)")
    ax.set_title("Where the liquidity lives (stylised estimate)")
    return ax


def plot_series(df: pd.DataFrame, title: str = "", ylabel: str = "EUR/MWh", ax=None, **kwargs):
    ax = ax or plt.gca()
    long = df.reset_index().melt(id_vars=df.index.name or "index", var_name="series", value_name="value")
    sns.lineplot(data=long, x=df.index.name or "index", y="value", hue="series", ax=ax, **kwargs)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    return ax


def plot_scatter_fit(x, y, beta0: float, beta1: float, xlabel: str, ylabel: str, ax=None, title: str = ""):
    ax = ax or plt.gca()
    sns.scatterplot(x=x, y=y, alpha=0.4, s=15, ax=ax)
    grid = np.linspace(np.min(x), np.max(x), 50)
    ax.plot(grid, beta0 + beta1 * grid, color="crimson", lw=2, label=f"fit: slope={beta1:.3f}")
    ax.axline((0, 0), slope=1, color="0.5", ls="--", lw=1, label="1:1 line")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return ax


def plot_pnl_distributions(pnls: dict[str, pd.Series], ax=None, bins: int = 60):
    ax = ax or plt.gca()
    long = pd.concat([pd.DataFrame({"pnl": v.values, "strategy": k}) for k, v in pnls.items()])
    sns.histplot(data=long, x="pnl", hue="strategy", bins=bins, element="step", stat="density",
                 common_norm=False, ax=ax)
    ax.set_title("Daily P&L distribution")
    ax.set_xlabel("EUR")
    return ax


def plot_heatmap(matrix: pd.DataFrame, title: str = "", ax=None, fmt: str = ".2f", **kwargs):
    ax = ax or plt.gca()
    sns.heatmap(matrix, annot=matrix.shape[0] <= 12, fmt=fmt, cmap="vlag", center=0, ax=ax, **kwargs)
    ax.set_title(title)
    return ax


def two_panels(figsize=(14, 4.5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    return fig, axes


__all__ = ["setup_style", "plot_curve", "plot_curve_with_strips", "plot_liquidity", "plot_series",
           "plot_scatter_fit", "plot_pnl_distributions", "plot_heatmap", "two_panels"]
