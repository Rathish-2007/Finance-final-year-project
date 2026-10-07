"""
Module 4 — TVP-VAR + QQC connectedness.

* Time-varying VAR(1) estimated on rolling windows → time-varying
  Generalised Forecast Error Variance Decomposition (Diebold-Yilmaz /
  Pesaran-Shin) → TCI(t), TO, FROM, NET.
* Quantile-conditional variant (QQC): each equation re-estimated by fast
  IRLS quantile regression at τ ∈ {0.05, 0.50, 0.95}, with the shock
  covariance rescaled by empirical tail dispersion, giving
  tail/median/boom connectedness — the regime-aware quantile-frequency
  layer that closes Gap G2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 10


# ------------------------------------------------------------------ fits
def fit_var_ols(Y: np.ndarray, lags: int = 1):
    T, n = Y.shape
    Yd = Y[lags:]
    Z = np.ones((T - lags, 1 + n * lags))
    for l in range(lags):
        Z[:, 1 + l * n:1 + (l + 1) * n] = Y[lags - 1 - l:T - 1 - l]
    coef, *_ = np.linalg.lstsq(Z, Yd, rcond=None)
    resid = Yd - Z @ coef
    dof = max(len(Yd) - Z.shape[1], 1)
    Sigma = resid.T @ resid / dof
    return coef, Sigma, Z, Yd, resid


def companion(coef: np.ndarray, n: int, lags: int):
    B = [coef[1 + l * n:1 + (l + 1) * n].T for l in range(lags)]
    if lags == 1:
        return B[0]
    A = np.zeros((n * lags, n * lags))
    for l in range(lags):
        A[:n, l * n:(l + 1) * n] = B[l]
    if lags > 1:
        A[n:, :-n] = np.eye(n * (lags - 1))
    return A


def _stabilise(A: np.ndarray, rho_max: float = 0.985):
    ev = np.abs(np.linalg.eigvals(A))
    rho = ev.max() if len(ev) else 0
    if rho > rho_max:
        A = A * (rho_max / rho)
    return A


def quantile_irls(Z: np.ndarray, y: np.ndarray, tau: float, iters: int = 8):
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    for _ in range(iters):
        r = y - Z @ beta
        w = 1.0 / np.maximum(np.abs(r), 1e-4)
        beta, *_ = np.linalg.lstsq(Z * w[:, None], y * w, rcond=None)
    return beta


def tail_scale(resid: np.ndarray, tau: float):
    """Dispersion rescaling k_i(τ) = E[r² | tail at τ] / E[r²] per variable."""
    n = resid.shape[1]
    k = np.ones(n)
    base = np.maximum((resid ** 2).mean(0), 1e-12)
    for i in range(n):
        r = resid[:, i]
        if tau < 0.5:
            q = np.quantile(r, tau)
            sel = r <= q
        elif tau > 0.5:
            q = np.quantile(r, tau)
            sel = r >= q
        else:
            sel = np.ones(len(r), bool)
        if sel.sum() > 5:
            k[i] = (r[sel] ** 2).mean() / base[i]
    return np.sqrt(np.clip(k, 0.25, 9.0))


# ------------------------------------------------------------------ FEVD
def generalised_fevd(A: np.ndarray, Sigma: np.ndarray, H: int = HORIZON):
    n = A.shape[0]
    A = _stabilise(A.copy())
    if Sigma.shape[0] < n:            # companion form: pad shock covariance
        S = np.zeros((n, n))
        S[:Sigma.shape[0], :Sigma.shape[1]] = Sigma
        Sigma = S
    num = np.zeros((n, n))
    den = np.zeros(n)
    Apow = np.eye(n)
    for _ in range(H):
        AS = Apow @ Sigma
        for i in range(n):
            den_i = AS[i, i]
            den[i] += max(den_i, 1e-14)
            for j in range(n):
                num[i, j] += AS[i, j] ** 2 / max(Sigma[j, j], 1e-14)
        Apow = Apow @ A
    fevd = num / den[:, None]
    fevd /= fevd.sum(1, keepdims=True)
    return fevd


def connectedness(fevd: np.ndarray):
    n = fevd.shape[0]
    off = fevd.copy()
    np.fill_diagonal(off, 0)
    tci = 100.0 * off.sum() / n
    to = 100.0 * off.sum(0)          # column j: what j transmits
    frm = 100.0 * off.sum(1)         # row i: what i receives
    net = to - frm
    return tci, to, frm, net


# ------------------------------------------------------------------ rolling
def rolling_qqc(returns: np.ndarray, dates, window: int = 252, step: int = 21,
                taus=(0.05, 0.50, 0.95), horizon: int = HORIZON,
                progress_cb=None):
    T, n = returns.shape
    out = {tau: {"dates": [], "tci": [], "fevd": None} for tau in taus}
    starts = list(range(0, T - window, step))
    for wi, t0 in enumerate(starts):
        Y = returns[t0:t0 + window]
        coef, Sigma, Z, Yd, resid = fit_var_ols(Y, lags=1)
        A_med = _stabilise(companion(coef, n, 1))
        for tau in taus:
            if tau == 0.5:
                A_tau = A_med
            else:
                coef_t = np.zeros_like(coef)
                coef_t[0] = coef[0]
                for j in range(n):
                    beta_q = quantile_irls(Z, Yd[:, j], tau, iters=6)
                    coef_t[:, j] = beta_q
                A_tau = _stabilise(companion(coef_t, n, 1))
            k = tail_scale(resid, tau)
            D = np.diag(k)
            Sigma_t = D @ Sigma @ D
            fevd = generalised_fevd(A_tau, Sigma_t, horizon)
            tci, *_ = connectedness(fevd)
            out[tau]["dates"].append(str(pd.Timestamp(dates[t0 + window]).date()))
            out[tau]["tci"].append(round(float(tci), 2))
            out[tau]["fevd"] = fevd        # keep last window for table/network
        if progress_cb and wi % 5 == 0:
            progress_cb(wi / len(starts))
    if progress_cb:
        progress_cb(1.0)

    summary = {}
    for tau in taus:
        fevd = out[tau]["fevd"]
        tci, to, frm, net = connectedness(fevd)
        summary[str(tau)] = {
            "tci": round(float(tci), 2),
            "to": [round(float(v), 2) for v in to],
            "from": [round(float(v), 2) for v in frm],
            "net": [round(float(v), 2) for v in net],
            "pairwise": [[round(float(fevd[i, j]), 3) for j in range(n)]
                         for i in range(n)],
        }
        out[tau].pop("fevd")
    return {"series": out, "summary": summary, "window": window, "step": step,
            "horizon": horizon}
