"""Price download (yfinance) and cleaning into an aligned return matrix. No Streamlit."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from warnings import catch_warnings, simplefilter

import pandas as pd
import yfinance as yf

DEFAULT_RF = 0.04
MAX_MISSING_FRAC = 0.05
FFILL_LIMIT = 5  # forward-fill at most this many consecutive missing days
MIN_OBS = 60
RESAMPLE_RULE = {"daily": None, "weekly": "W-FRI", "monthly": "ME"}


class DataError(RuntimeError):
    """Price data could not be retrieved (network failure or nothing returned)."""


@dataclass
class CleanData:
    prices: pd.DataFrame
    returns: pd.DataFrame
    dropped: dict[str, str] = field(default_factory=dict)  # ticker -> reason
    warnings: list[str] = field(default_factory=list)
    start: date | None = None  # first and last trading day actually used
    end: date | None = None


def parse_tickers(text: str) -> list[str]:
    """Split on commas/whitespace/semicolons, uppercase, de-duplicate keeping order."""
    seen: dict[str, None] = {}
    for tok in re.split(r"[,;\s]+", text or ""):
        tok = tok.strip().upper()
        if tok:
            seen.setdefault(tok, None)
    return list(seen)


def download_prices(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Adjusted close prices, one column per requested ticker (all-NaN if not found)."""
    try:
        with catch_warnings():  # yfinance emits pandas deprecation noise
            simplefilter("ignore")
            raw = yf.download(
                tickers,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),  # yfinance's end is exclusive
                auto_adjust=True,
                progress=False,
                threads=True,
            )
    except Exception as exc:  # yfinance surfaces network problems as assorted exceptions
        raise DataError(f"Could not download prices from Yahoo Finance: {exc}") from exc
    if raw is None or raw.empty:
        raise DataError(
            "Yahoo Finance returned no data. Check the tickers and your internet connection, "
            "or try again in a minute (Yahoo sometimes rate-limits)."
        )
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].set_axis(tickers, axis=1)
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.reindex(columns=tickers)


def fetch_risk_free_rate() -> float | None:
    """Latest 13-week T-bill yield (^IRX, quoted in percent) as a decimal, or None on failure."""
    try:
        with catch_warnings():
            simplefilter("ignore")
            raw = yf.download("^IRX", period="1mo", auto_adjust=True, progress=False)
        close = raw["Close"].squeeze().dropna()
        if close.empty:
            return None
        return float(close.iloc[-1]) / 100.0
    except Exception:
        return None


def clean_prices(raw: pd.DataFrame, requested_start: date, freq: str = "daily") -> CleanData:
    """Align prices on a common window, drop bad assets, resample, and compute simple returns."""
    prices = raw.copy()
    dropped: dict[str, str] = {}
    warnings: list[str] = []

    for t in prices.columns[prices.notna().sum() < 2]:
        dropped[t] = "no price data found (invalid, delisted, or not on Yahoo Finance)"
        if "." in t:
            dropped[t] += f". If it's a share class, Yahoo uses a dash: {t.replace('.', '-')}"
    prices = prices.drop(columns=list(dropped))
    if prices.empty:
        return CleanData(prices, prices, dropped, warnings)

    # 24/7 assets (crypto) also trade on weekends and exchange holidays. Use the other assets'
    # calendar so stocks don't get fake 0% returns on days their market was closed.
    weekend = prices.index.dayofweek >= 5
    round_the_clock = prices.columns[prices[weekend].notna().any()]
    regular = prices.columns.difference(round_the_clock)
    prices = prices[prices[regular].notna().any(axis=1)] if len(regular) else prices[~weekend]
    prices = prices.dropna(how="all")

    # Drop assets that stopped trading or are gappy over their own history, *before* setting
    # the common window, so a dropped asset can't still shorten it.
    first = prices.apply(pd.Series.first_valid_index)
    last = prices.apply(pd.Series.last_valid_index)
    for t in prices.columns:
        trailing_gap = int((prices.index > last[t]).sum())
        missing = prices.loc[first[t]:, t].isna().mean()
        if trailing_gap > FFILL_LIMIT:
            dropped[t] = f"stopped trading on {last[t]:%Y-%m-%d} (possibly delisted, merged or renamed)"
        elif missing > MAX_MISSING_FRAC:
            dropped[t] = f"{missing:.0%} of days missing (limit {MAX_MISSING_FRAC:.0%})"
    prices = prices.drop(columns=[t for t in prices.columns if t in dropped])
    if prices.empty:
        return CleanData(prices, prices, dropped, warnings)

    # Common window starts when the youngest asset starts trading.
    first = prices.apply(pd.Series.first_valid_index)
    window_start = first.max()
    late = first[first > pd.Timestamp(requested_start) + pd.Timedelta(days=30)]
    if not late.empty:
        names = ", ".join(f"{t} (from {d:%Y-%m-%d})" for t, d in late.sort_values().items())
        warnings.append(
            f"Short price history shortens the analysis window for everyone: {names}. "
            "Remove these tickers to use the full lookback."
        )
    # ...and ends on the last day *every* asset has a price, so the newest bar isn't forward-filled
    # into a fake 0% return (e.g. today's stock close is in but today's crypto bar isn't).
    window_end = prices.apply(pd.Series.last_valid_index).min()
    daily = prices.loc[window_start:window_end].ffill(limit=FFILL_LIMIT).dropna()
    if daily.empty:
        return CleanData(daily, daily, dropped, warnings)

    prices = daily
    rule = RESAMPLE_RULE[freq]
    if rule:
        prices = daily.resample(rule).last().dropna()
        label = prices.index[-1]
        period_last_bday = label if label.dayofweek < 5 else label - pd.offsets.BDay(1)
        if daily.index[-1] < period_last_bday:  # final week/month still in progress
            prices = prices.iloc[:-1]
        if prices.empty:
            return CleanData(prices, prices, dropped, warnings)
    # Actual trading days behind the first and last prices used (resample labels are period ends).
    start = daily.index[daily.index <= prices.index[0]][-1].date()
    end = daily.index[daily.index <= prices.index[-1]][-1].date()
    returns = prices.pct_change().dropna()

    if 0 < len(returns) < MIN_OBS:
        warnings.append(
            f"Only {len(returns)} {freq} return observations. Estimates will be very noisy; "
            "use a longer lookback or a higher frequency."
        )
    return CleanData(prices, returns, dropped, warnings, start, end)
