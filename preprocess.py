"""Module 1b — Preprocessing: log-returns + volatility normalisation (Raddant-style)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all").fillna(0.0)


def ewma_vol(returns: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    return returns.ewm(span=span, min_periods=max(5, span // 4)).std()


def standardise(returns: pd.DataFrame, span: int = 20, clip: float = 5.0) -> pd.DataFrame:
    """GARCH-style volatility normalisation (EWMA RiskMetrics approximation)."""
    vol = ewma_vol(returns, span).replace(0.0, np.nan)
    z = (returns / vol).clip(-clip, clip).fillna(0.0)
    return z


def winsorise(series: np.ndarray, lo: float = 0.01, hi: float = 0.99) -> np.ndarray:
    ql, qh = np.quantile(series, [lo, hi])
    return np.clip(series, ql, qh)
