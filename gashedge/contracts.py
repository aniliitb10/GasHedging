"""
Contract mechanics for European gas futures (TTF / PEG style monthly futures).

Everything here is *mechanical* (calendars, codes, hour counts) and can be
verified against the exchange product specification.  Where the exchange
rule depends on a holiday calendar we approximate with weekdays only and say so.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from gashedge.logging_config import get_logger

log = get_logger(__name__)

# Industry-standard futures month codes (identical across ICE, CME, EEX).
MONTH_CODES = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
               7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}
CODE_TO_MONTH = {v: k for k, v in MONTH_CODES.items()}

# Gas delivery is settled in local (CET/CEST) time, so delivery hours change with DST.
DELIVERY_TZ = ZoneInfo("Europe/Amsterdam")

# Gas day convention in continental Europe: 06:00 to 06:00 local time.
GAS_DAY_START_HOUR = 6


@dataclass(frozen=True)
class ContractSpec:
    """Stylised spec of an ICE Endex style monthly gas future (verify against the exchange)."""

    root: str = "TFM"                 # ICE code for Dutch TTF Gas Futures
    hub: str = "TTF"
    currency: str = "EUR"
    price_unit: str = "EUR/MWh"
    lot_size_mw: float = 1.0          # 1 MW delivered every hour of the delivery period
    tick_size: float = 0.005          # EUR/MWh
    expiry_business_days_before_delivery: int = 2


TTF_SPEC = ContractSpec()
PEG_SPEC = ContractSpec(root="PEG", hub="PEG")  # illustrative root; verify with the venue
NBP_SPEC = ContractSpec(root="NBP", hub="NBP", currency="GBP", price_unit="GBp/therm", tick_size=0.005)


def month_code(month: int) -> str:
    return MONTH_CODES[month]


def contract_symbol(year: int, month: int, root: str = "TFM") -> str:
    """E.g. contract_symbol(2026, 12) -> 'TFMZ26'."""
    return f"{root}{MONTH_CODES[month]}{year % 100:02d}"


def parse_symbol(symbol: str) -> tuple[str, int, int]:
    """Inverse of contract_symbol; assumes 20xx years. Returns (root, year, month)."""
    root, code, yy = symbol[:-3], symbol[-3], int(symbol[-2:])
    return root, 2000 + yy, CODE_TO_MONTH[code]


def delivery_period(year: int, month: int) -> tuple[datetime, datetime]:
    """Start/end of the delivery month measured in gas days (06:00 local to 06:00 local)."""
    start = datetime(year, month, 1, GAS_DAY_START_HOUR, tzinfo=DELIVERY_TZ)
    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    end = datetime(ny, nm, 1, GAS_DAY_START_HOUR, tzinfo=DELIVERY_TZ)
    return start, end


def delivery_hours(year: int, month: int) -> int:
    """
    Number of hours in the delivery month, DST aware.
    March (clocks forward) has 743 hours, October (clocks back) has 745, others 24 * days.
    """
    start, end = delivery_period(year, month)
    hours = (end.astimezone(ZoneInfo("UTC")) - start.astimezone(ZoneInfo("UTC"))).total_seconds() / 3600
    return int(round(hours))


def lot_mwh(year: int, month: int, spec: ContractSpec = TTF_SPEC) -> float:
    """Energy (MWh) represented by one lot: lot_size_mw * delivery hours."""
    return spec.lot_size_mw * delivery_hours(year, month)


def expiry_date(year: int, month: int, spec: ContractSpec = TTF_SPEC) -> date:
    """
    Last trading day: N business days before the first calendar day of the delivery month.
    Weekday approximation only - the exchange applies its own holiday calendar.
    """
    first = np.datetime64(date(year, month, 1))
    last = np.busday_offset(first, -spec.expiry_business_days_before_delivery, roll="backward")
    return last.astype("datetime64[D]").astype(date)


# ---------------------------------------------------------------- strips ----

def quarter_months(year: int, q: int) -> list[tuple[int, int]]:
    return [(year, m) for m in range(3 * (q - 1) + 1, 3 * q + 1)]


def season_months(year: int, season: str) -> list[tuple[int, int]]:
    """Summer = Apr..Sep of `year`; Winter = Oct..Dec of `year` + Jan..Mar of `year`+1."""
    if season.lower().startswith("sum"):
        return [(year, m) for m in range(4, 10)]
    return [(year, m) for m in (10, 11, 12)] + [(year + 1, m) for m in (1, 2, 3)]


def calendar_months(year: int) -> list[tuple[int, int]]:
    return [(year, m) for m in range(1, 13)]


def strip_label(kind: str, year: int, index: int | str | None = None) -> str:
    if kind == "Q":
        return f"Q{index}-{year % 100:02d}"
    if kind == "SUM":
        return f"Sum-{year % 100:02d}"
    if kind == "WIN":
        return f"Win-{year % 100:02d}"
    return f"Cal-{year % 100:02d}"


def strip_price(monthly_prices: dict[tuple[int, int], float], months: list[tuple[int, int]]) -> float:
    """
    Fair strip price = hours-weighted average of its monthly legs.

        P_strip = sum_i H_i * P_i / sum_i H_i

    because one lot of a strip delivers 1 MW in *every* hour of every month.
    """
    hours = np.array([delivery_hours(y, m) for y, m in months], dtype=float)
    prices = np.array([monthly_prices[(y, m)] for y, m in months], dtype=float)
    return float(np.dot(hours, prices) / hours.sum())


def listed_months(as_of: date, n_months: int) -> list[tuple[int, int]]:
    """The next `n_months` delivery months whose expiry has not passed at `as_of`."""
    y, m = as_of.year, as_of.month
    out: list[tuple[int, int]] = []
    while len(out) < n_months:
        if expiry_date(y, m) >= as_of:
            out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def contract_table(as_of: date, n_months: int = 36, spec: ContractSpec = TTF_SPEC) -> pd.DataFrame:
    """One row per listed monthly contract with symbol, hours, lot MWh and expiry."""
    rows = []
    for i, (y, m) in enumerate(listed_months(as_of, n_months), start=1):
        rows.append({
            "tenor": i,
            "year": y,
            "month": m,
            "symbol": contract_symbol(y, m, spec.root),
            "delivery_start": date(y, m, 1),
            "hours": delivery_hours(y, m),
            "lot_mwh": lot_mwh(y, m, spec),
            "expiry": expiry_date(y, m, spec),
            "days_to_expiry": (expiry_date(y, m, spec) - as_of).days,
        })
    df = pd.DataFrame(rows)
    log.debug("Built contract table with %d rows as of %s", len(df), as_of)
    return df


def next_delivery_month(as_of: date) -> tuple[int, int]:
    return listed_months(as_of, 1)[0]


def add_months(year: int, month: int, k: int) -> tuple[int, int]:
    idx = (year * 12 + (month - 1)) + k
    return idx // 12, idx % 12 + 1


def days_between(a: date, b: date) -> int:
    return (b - a).days


__all__ = [
    "MONTH_CODES", "CODE_TO_MONTH", "ContractSpec", "TTF_SPEC", "PEG_SPEC", "NBP_SPEC",
    "month_code", "contract_symbol", "parse_symbol", "delivery_period", "delivery_hours",
    "lot_mwh", "expiry_date", "quarter_months", "season_months", "calendar_months",
    "strip_label", "strip_price", "listed_months", "contract_table", "next_delivery_month",
    "add_months", "days_between", "timedelta",
]
