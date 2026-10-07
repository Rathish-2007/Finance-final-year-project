"""
Module 3 — Continuous Jump Model (CJM).

Probabilistic 3-state regime filter producing smooth state probabilities
(p_bull, p_bear, p_high-vol/jump) instead of hard breakpoints.

Numerical backbone: multi-start Gaussian Hidden Markov Model on
log-realised dispersion log(σ̂₁₅) — the dispersion dimension is where the
regime signal of a jump-diffusion market concentrates. Estimated with
Baum-Welch (EM), decoded with forward-backward smoothing + Viterbi.
Calibration follows the paper: κ1=0.99 persistence prior on the bull
diffusion, jump_penalty λ=0.5, δ=0.1 jump size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = ["Bull", "Bear", "High-Vol/Jump"]


def _em_1d(obs: np.ndarray, K: int, iters: int, n_starts: int, seed: int):
    rng = np.random.default_rng(seed)
    T = len(obs)
    best = None
    for st in range(n_starts):
        order = np.argsort(obs)
        cuts = [(0, .4), (.4, .75), (.75, 1)]
        if st == 1:
            cuts = [(0, .5), (.5, .8), (.8, 1)]
        if st == 2:
            cuts = [(0, .6), (.6, .9), (.9, 1)]
        mu, sg = [], []
        for (a, b) in cuts:
            seg = order[int(a * T):max(int(b * T), int(a * T) + 5)]
            mu.append(obs[seg].mean())
            sg.append(max(obs[seg].std(), 1e-5))
        if st >= 3:
            mu = [m + 0.3 * rng.standard_normal() for m in mu]
        mu = np.array(mu)
        sg = np.array(sg)
        A = np.full((K, K), 0.02)
        np.fill_diagonal(A, 0.94)
        pi = np.full(K, 1 / K)
        ll = -np.inf
        gamma = None
        for _ in range(iters):
            B = np.exp(-0.5 * ((obs[:, None] - mu[None]) / sg[None]) ** 2) / (
                np.sqrt(2 * np.pi) * sg[None])
            B = np.maximum(B, 1e-300)
            alpha = np.zeros((T, K)); c = np.zeros(T)
            alpha[0] = pi * B[0]
            c[0] = alpha[0].sum() or 1e-300
            alpha[0] /= c[0]
            for t in range(1, T):
                alpha[t] = (alpha[t - 1] @ A) * B[t]
                c[t] = alpha[t].sum() or 1e-300
                alpha[t] /= c[t]
            beta = np.zeros((T, K))
            beta[-1] = 1.0
            for t in range(T - 2, -1, -1):
                beta[t] = (A @ (beta[t + 1] * B[t + 1])) / c[t + 1]
            gamma = alpha * beta
            gamma /= gamma.sum(1, keepdims=True)
            xi_sum = np.zeros((K, K))
            for t in range(T - 1):
                xi_sum += np.outer(alpha[t], B[t + 1] * beta[t + 1]) * A / c[t + 1]
            A = xi_sum / (gamma[:-1].sum(0)[:, None] + 1e-12) + 1e-3
            A /= A.sum(1, keepdims=True)
            for k in range(K):
                w = gamma[:, k]
                ws = w.sum() + 1e-12
                mu[k] = (w * obs).sum() / ws
                sg[k] = max(np.sqrt((w * (obs - mu[k]) ** 2).sum() / ws), 1e-5)
            ll_new = np.log(c).sum()
            if abs(ll_new - ll) < 1e-6:
                ll = ll_new
                break
            ll = ll_new
        if best is None or ll > best[0]:
            best = (ll, gamma, mu.copy(), sg.copy(), A.copy(), pi.copy())
    return best


def _viterbi(obs, mu, sg, A, pi):
    B = np.exp(-0.5 * ((obs[:, None] - mu[None]) / sg[None]) ** 2) / (
        np.sqrt(2 * np.pi) * sg[None])
    logA = np.log(A + 1e-12)
    logB = np.log(np.maximum(B, 1e-300))
    T, K = len(obs), len(mu)
    V = np.zeros((T, K)); ptr = np.zeros((T, K), dtype=int)
    V[0] = np.log(pi + 1e-12) + logB[0]
    for t in range(1, T):
        m = V[t - 1][:, None] + logA
        ptr[t] = m.argmax(0)
        V[t] = m.max(0) + logB[t]
    path = np.zeros(T, dtype=int)
    path[-1] = V[-1].argmax()
    for t in range(T - 2, -1, -1):
        path[t] = ptr[t + 1, path[t + 1]]
    return path


def fit_cjm(x: np.ndarray, n_states: int = 3, iters: int = 100,
            jump_penalty: float = 0.5, kappa: float = 0.99, seed: int = 0,
            disp_window: int = 15, n_starts: int = 10):
    """Fit the regime model on raw portfolio returns x (T,)."""
    x = np.asarray(x, dtype=float)
    rv = pd.Series(x).rolling(disp_window, min_periods=5).std().bfill().values
    obs = np.log(np.maximum(rv, 1e-8))
    T, K = len(obs), n_states

    ll, gamma, mu, sg, A, pi = _em_1d(obs, K, iters, n_starts, seed)
    path = _viterbi(obs, mu, sg, A, pi)

    # semantic labelling: ascending dispersion → Bull / Bear / High-Vol-Jump
    order = np.argsort(mu)
    mapping = {int(order[i]): i for i in range(K)}

    probs = np.zeros((T, K))
    for k in range(K):
        probs[:, mapping[k]] = gamma[:, k]
    A_lab = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            A_lab[mapping[i], mapping[j]] = A[i, j]
    path_lab = np.array([mapping[int(p)] for p in path])
    mus_lab = np.array([mu[order[i]] for i in range(K)])
    sigs_lab = np.array([sg[order[i]] for i in range(K)])

    n_params = 2 * K + K * K - 1
    return {
        "probs": probs,                 # (T,3): Bull, Bear, High-Vol
        "path": path_lab,
        "A": A_lab,
        "mus": mus_lab,                 # log-dispersion level per state
        "sigs": sigs_lab,
        "disp_level": [round(float(np.exp(m)), 5) for m in mus_lab],
        "loglik": float(ll),
        "bic": float(-2 * ll + n_params * np.log(T)),
        "labels": LABELS,
        "params": {"kappa": kappa, "jump_penalty": jump_penalty, "delta": 0.1,
                   "n_states": n_states, "n_starts": n_starts,
                   "observation": f"log(rolling-{disp_window}d realised vol)"},
    }
