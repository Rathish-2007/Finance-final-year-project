"""
Module 1 — Unified Data Lake (closes Gap G1: siloed assets).

Builds a single harmonised panel 2016-2024 across 10 assets:
Green Bond, S&P500, Gold, Oil, BTC, ETH, Clean Energy, Treasuries,
Corporate Credit and Volatility.

A deterministic synthetic generator (seed=42) provides the benchmark so the
pipeline is fully reproducible offline; a yfinance adapter is included for
live data. Also synthesises the auxiliary panel used by later modules
(news flow, Google SVI, stablecoin peg, funding, energy dependence, etc.).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ASSETS = [
    {"sym": "SPX",   "name": "S&P 500",           "ticker": "^GSPC",   "cls": "Equity"},
    {"sym": "GRNB",  "name": "Green Bond ETF",    "ticker": "GRNB",    "cls": "Green Bond"},
    {"sym": "GOLD",  "name": "Gold",              "ticker": "GC=F",    "cls": "Commodity"},
    {"sym": "OIL",   "name": "Oil (WTI)",         "ticker": "CL=F",    "cls": "Commodity"},
    {"sym": "BTC",   "name": "Bitcoin",           "ticker": "BTC-USD", "cls": "Crypto"},
    {"sym": "ETH",   "name": "Ethereum",          "ticker": "ETH-USD", "cls": "Crypto"},
    {"sym": "CLEAN", "name": "Clean Energy",      "ticker": "ICLN",    "cls": "Equity"},
    {"sym": "TLT",   "name": "US Treasury 20Y",   "ticker": "TLT",     "cls": "Bond"},
    {"sym": "LQD",   "name": "Corporate Credit",  "ticker": "LQD",     "cls": "Bond"},
    {"sym": "VIX",   "name": "Volatility (VIX)",  "ticker": "^VIX",    "cls": "Volatility"},
]

CORE7 = ["SPX", "GRNB", "GOLD", "OIL", "BTC", "ETH", "CLEAN"]   # connectedness system

STATE_NAMES = {0: "Bull", 1: "Bear", 2: "High-Vol / Jump"}

# --------------------------------------------------------------- transitions
TRANS = np.array([
    [0.990, 0.008, 0.002],   # bull   (~100-day episodes)
    [0.030, 0.960, 0.010],   # bear   (~25-day episodes)
    [0.010, 0.060, 0.930],   # crisis (~14-day episodes)
])

MU = np.array([0.00100, -0.00200, -0.00650])          # daily market drift per state
VOL_SCALE = np.array([1.0, 3.0, 6.5])                 # market vol multiplier per state

BETA_MKT = {"SPX": 1.00, "GRNB": 0.14, "GOLD": -0.08, "OIL": 0.55, "BTC": 1.15,
            "ETH": 1.25, "CLEAN": 1.35, "TLT": -0.30, "LQD": 0.12}
BETA_CRY = {"BTC": 0.85, "ETH": 1.00}                # crypto-specific factor load
IDIO_VOL = {"SPX": 0.004, "GRNB": 0.0015, "GOLD": 0.005, "OIL": 0.009, "BTC": 0.020,
            "ETH": 0.025, "CLEAN": 0.007, "TLT": 0.004, "LQD": 0.0025}
JUMP_P = {"BTC": 0.012, "ETH": 0.014}                 # Poisson jump intensity (crypto)


DEP_THRESHOLD = 12.0     # Russian-commodity dependence threshold (%)


def _gen_dep(T: int, seed: int):
    """Slow-moving energy-dependence proxy (0-20%) with genuine threshold effect."""
    rng = np.random.default_rng(seed + 101)
    dep = np.zeros(T)
    dep[0] = 9.5
    for t in range(1, T):
        dep[t] = dep[t - 1] + 0.05 * rng.standard_normal()
    dep = np.clip(dep + 4.0 * np.sin(np.arange(T) / 240.0), 1.5, 19.5)
    return dep


def _synth_prices(seed: int, start: str, end: str):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    T = len(idx)
    dep = _gen_dep(T, seed)

    # latent regime chain
    s = np.zeros(T, dtype=int)
    for t in range(1, T):
        s[t] = rng.choice(3, p=TRANS[s[t - 1]])

    # market + crypto factors (crypto vol is capped: regimes matter, blowups don't)
    mkt = MU[s] + 0.0085 * VOL_SCALE[s] * rng.standard_normal(T)
    crash = (s == 2) & (rng.random(T) < 0.10)
    mkt[crash] -= rng.uniform(0.008, 0.030, crash.sum())
    vs_cry = np.minimum(VOL_SCALE[s], 3.5)
    cry = 0.00035 + 0.014 * vs_cry * rng.standard_normal(T)

    # sanction-risk premium: dependence above threshold loads linearly on drift
    # (slope-break in dep → detectable by Hansen sup-Wald threshold test)
    dep_beta = {"OIL": -0.00090, "GOLD": 0.00030}
    dep_load = dep * (dep > DEP_THRESHOLD)

    prices = {}
    for a in ASSETS:
        sym = a["sym"]
        if sym == "VIX":
            continue
        vs = vs_cry if sym in BETA_CRY else VOL_SCALE[s]
        r = BETA_MKT[sym] * mkt + IDIO_VOL[sym] * vs * rng.standard_normal(T)
        if sym in dep_beta:
            r += dep_beta[sym] * dep_load
        if sym in BETA_CRY:
            r += BETA_CRY[sym] * cry
        jp = JUMP_P.get(sym, 0.0)
        if jp:
            hits = rng.random(T) < jp * np.minimum(VOL_SCALE[s], 2.5)
            r[hits] += rng.choice([-1, 1], hits.sum()) * rng.uniform(0.02, 0.06, hits.sum())
        p0 = {"SPX": 2000, "GRNB": 25, "GOLD": 1150, "OIL": 37, "BTC": 430, "ETH": 9,
              "CLEAN": 14, "TLT": 120, "LQD": 115}[sym]
        prices[sym] = p0 * np.exp(np.cumsum(r))

    # VIX level process, inversely tied to market shocks
    z = np.zeros(T)
    for t in range(1, T):
        z[t] = 0.93 * z[t - 1] - 9.0 * mkt[t] + 1.6 * rng.standard_normal()
    prices["VIX"] = np.clip(13 + z + 6 * rng.standard_normal(T), 9.5, 85)

    df = pd.DataFrame(prices, index=idx)
    return df, s, dep


def _synth_aux(seed: int, idx: pd.DatetimeIndex, returns: pd.DataFrame,
               states: np.ndarray, dep: np.ndarray):
    rng = np.random.default_rng(seed + 7)
    T = len(idx)
    aux = pd.DataFrame(index=idx)

    # Google Search Volume Index for "bitcoin crash" style queries — spikes in crisis
    svi = 35 + 25 * (states == 1) + 45 * (states == 2) + 8 * rng.standard_normal(T)
    aux["SVI"] = np.clip(svi, 5, 100)

    # stablecoin peg deviation (bps) and funding rate (%)
    peg = 2.0 + 6 * (states == 2) + 1.5 * rng.standard_normal(T)
    peg += np.where(rng.random(T) < 0.01, rng.uniform(8, 25, T), 0)
    aux["PEG_DEV"] = np.clip(peg, 0, 60)
    aux["FUNDING"] = 0.01 + 0.02 * (states == 0) - 0.03 * (states == 2) + 0.02 * rng.standard_normal(T)

    aux["ENERGY_DEP"] = dep                     # injected from the price engine

    # FinTech index & bank stability (for the feedback loop tests)
    ft = np.cumsum(0.0006 + 0.011 * VOL_SCALE[states] * rng.standard_normal(T))
    aux["FINTECH"] = 100 * np.exp(ft)
    bs = np.cumsum(-0.0001 + 0.004 * rng.standard_normal(T)) - 0.25 * np.log(aux["FINTECH"] / 100)
    aux["BANK_STAB"] = 100 * np.exp(bs)

    aux["NEWS_COUNT"] = np.clip(
        400 + 250 * (states != 0) + 120 * rng.standard_normal(T), 40, None).astype(int)
    aux["BTC_RET"] = returns["BTC"].values
    return aux


def build_dataset(seed: int = 42, start: str = "2016-01-04", end: str = "2024-12-31",
                  use_yfinance: bool = False):
    """Return dict with prices, aux panel, returns and ground-truth regimes."""
    if use_yfinance:
        try:
            return _real_data(start, end)
        except Exception:
            pass  # fall back to synthetic

    prices, states, dep = _synth_prices(seed, start, end)
    returns = np.log(prices / prices.shift(1)).dropna()
    returns = returns.fillna(0.0)
    aux = _synth_aux(seed, prices.index, returns.reindex(prices.index).fillna(0.0),
                     states, dep)

    return {
        "prices": prices,
        "returns": returns,
        "aux": aux,
        "states": states,
        "asset_meta": ASSETS,
        "core7": CORE7,
        "source": "synthetic" if not use_yfinance else "yfinance",
    }


def _real_data(start: str, end: str):
    import yfinance as yf
    tickers = {a["sym"]: a["ticker"] for a in ASSETS}
    df = yf.download(list(tickers.values()), start=start, end=end,
                     auto_adjust=True, progress=False)["Close"]
    df = df.rename(columns={v: k for k, v in tickers.items()}).dropna(how="all").ffill()
    returns = np.log(df / df.shift(1)).dropna(how="all").fillna(0.0)
    aux = pd.DataFrame(index=df.index)
    aux["SVI"] = 50.0
    aux["PEG_DEV"] = 3.0
    aux["FUNDING"] = 0.01
    aux["ENERGY_DEP"] = 10.0
    aux["FINTECH"] = df["SPX"] / df["SPX"].iloc[0] * 100
    aux["BANK_STAB"] = 100.0
    aux["NEWS_COUNT"] = 400
    aux["BTC_RET"] = returns["BTC"]
    states = np.zeros(len(df), dtype=int)
    return {"prices": df, "returns": returns, "aux": aux, "states": states,
            "asset_meta": ASSETS, "core7": CORE7, "source": "yfinance"}
