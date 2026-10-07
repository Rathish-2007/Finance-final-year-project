"""
Module 5 — Threshold / feedback / stability laboratory (closes Gap G5).

* Hansen (2000) sup-Wald threshold test with wild bootstrap — tests whether
  Russian-commodity dependence enters returns non-linearly (Lo 2022).
* Granger-causality battery for the FinTech ↔ bank-stability feedback loop
  (Goldstein et al.) and attention (SVI) → crypto dynamics.
* VAR spillover analysis of stablecoin run risk (Joebges et al.):
  [peg deviation, BTC return, funding rate] system.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import tvpvar

warnings.filterwarnings("ignore")


# ------------------------------------------------------------- Hansen test
def _ols_ssr(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return float(r @ r)


def hansen_threshold(y: np.ndarray, q: np.ndarray, control: np.ndarray,
                     lo: float = 0.15, hi: float = 0.85, n_grid: int = 40,
                     n_boot: int = 300, seed: int = 3):
    """sup-Wald threshold test:  y = a + b·control + β1 q·1(q≤γ) + β2 q·1(q>γ)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    q = np.asarray(q, float)
    control = np.asarray(control, float)
    grid = np.unique(np.quantile(q, np.linspace(lo, hi, n_grid)))

    def sup_f(yv):
        X0 = np.column_stack([np.ones(len(yv)), control, q])
        ssr0 = _ols_ssr(yv, X0)
        best, best_ssr1 = 0.0, ssr0
        for g in grid:
            d = (q > g).astype(float)
            X1 = np.column_stack([np.ones(len(yv)), control, q * (1 - d), q * d])
            ssr1 = _ols_ssr(yv, X1)
            if ssr0 - ssr1 > best:
                best, best_ssr1 = ssr0 - ssr1, ssr1
        sig2 = max(best_ssr1 / max(len(yv) - 5, 1), 1e-12)
        return best / sig2

    F_obs = sup_f(y)

    # wild (Rademacher) bootstrap under H0
    X0 = np.column_stack([np.ones(len(y)), control, q])
    beta0, *_ = np.linalg.lstsq(X0, y, rcond=None)
    resid = y - X0 @ beta0
    count = 0
    for _ in range(n_boot):
        e = resid * rng.choice([-1.0, 1.0], len(resid))
        if sup_f(X0 @ beta0 + e) >= F_obs:
            count += 1

    best_g, best_f = None, -1
    X0 = np.column_stack([np.ones(len(y)), control, q])
    ssr0 = _ols_ssr(y, X0)
    for g in grid:
        d = (q > g).astype(float)
        X1 = np.column_stack([np.ones(len(y)), control, q * (1 - d), q * d])
        f = ssr0 - _ols_ssr(y, X1)
        if f > best_f:
            best_f, best_g = f, g
    return {"F_stat": round(float(F_obs), 3),
            "p_value": round((count + 1) / (n_boot + 1), 4),
            "threshold": round(float(best_g), 3),
            "n_boot": n_boot,
            "significant": (count + 1) / (n_boot + 1) < 0.10}


# ------------------------------------------------------------- Granger tests
def granger_battery(df: pd.DataFrame, pairs, maxlag: int = 5):
    import contextlib, io
    from statsmodels.tsa.stattools import grangercausalitytests
    out = []
    for a, b in pairs:
        x = df[[b, a]].copy()
        x = (x - x.mean()) / (x.std() + 1e-12)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                res = grangercausalitytests(x, maxlag=maxlag)
            pvals = {lag: res[lag][0]["ssr_ftest"][1] for lag in res}
            best_lag = min(pvals, key=pvals.get)
            out.append({"cause": a, "effect": b, "lag": int(best_lag),
                        "p_value": round(float(pvals[best_lag]), 4),
                        "significant": float(pvals[best_lag]) < 0.05})
        except Exception:       # singular matrix etc.
            out.append({"cause": a, "effect": b, "lag": None,
                        "p_value": None, "significant": False})
    return out


# ------------------------------------------------------------- stablecoin VAR
def stablecoin_run_risk(peg_dev: np.ndarray, btc_ret: np.ndarray,
                        funding: np.ndarray, lags: int = 2, horizon: int = 5):
    Y = np.column_stack([peg_dev, btc_ret, funding])
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)
    coef, Sigma, *_ = tvpvar.fit_var_ols(Y, lags=lags)
    n = Y.shape[1]
    A = tvpvar._stabilise(tvpvar.companion(coef, n, lags))
    fevd = tvpvar.generalised_fevd(A, Sigma, horizon)
    tci, to, frm, net = tvpvar.connectedness(fevd)
    names = ["Peg deviation", "BTC return", "Funding rate"]
    return {
        "names": names,
        "fevd": [[round(float(fevd[i, j]), 3) for j in range(n)] for i in range(n)],
        "tci": round(float(tci), 2),
        "to": [round(float(v), 2) for v in to],
        "from": [round(float(v), 2) for v in frm],
        "net": [round(float(v), 2) for v in net],
        "lags": lags, "horizon": horizon,
    }
