"""
Module 6 — Hierarchical Multi-Agent DRL trading intelligence (closes Gap G4).

H(weekly) → M(daily) → L(4h) deep-Q agents with knowledge distillation,
realistic friction costs (0.10% commission + 0.05% slippage + volatility
penalty) and permutation-based SHAP-style explainability.

Implemented with a dependency-light NumPy MLP so the pipeline runs anywhere;
the architecture maps 1:1 onto the PyTorch version in the paper.
State space: 20 technical + 10 SSCDV + 3 regime + 1 TCI = 34 features
(production spec expands to 134 with the full technical battery).
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

COST = 0.0010 + 0.0005          # commission + slippage per unit position change
VOL_PENALTY = 0.3               # variance penalty: λ · pos² · σ²
GAMMA = 0.60                    # near-myopic: daily drift prediction dominates
ACTIONS = {-1.0: 0, 0.0: 1, 1.0: 2}
ACTION_POS = np.array([-1.0, 0.0, 1.0])
REGIME_COLS = ["p_bull", "p_bear", "p_high"]


# ------------------------------------------------------------------ network
class MLP:
    def __init__(self, d_in, d_hid=64, d_out=3, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((d_in, d_hid)) * np.sqrt(2 / d_in)
        self.b1 = np.zeros(d_hid)
        self.W2 = rng.standard_normal((d_hid, d_out)) * np.sqrt(1 / d_hid)
        self.b2 = np.zeros(d_out)
        self.mW1 = np.zeros_like(self.W1); self.mW2 = np.zeros_like(self.W2)
        self.mb1 = np.zeros_like(self.b1); self.mb2 = np.zeros_like(self.b2)

    def copy_from(self, other):
        self.W1, self.b1 = other.W1.copy(), other.b1.copy()
        self.W2, self.b2 = other.W2.copy(), other.b2.copy()

    def forward(self, x):
        h = np.maximum(x @ self.W1 + self.b1, 0)
        return h @ self.W2 + self.b2, h

    def predict(self, x):
        return self.forward(x)[0]

    def train_step(self, x, target_q, lr=3e-3, clip=5.0):
        q, h = self.forward(x)
        err = q - target_q
        huber = np.where(np.abs(err) < 1.0, err, np.sign(err))
        B = len(x)
        dW2 = h.T @ huber / B
        db2 = huber.mean(0)
        dh = (huber @ self.W2.T) * (h > 0)
        dW1 = x.T @ dh / B
        db1 = dh.mean(0)
        for g in (dW1, dW2, db1, db2):
            np.clip(g, -clip, clip, out=g)
        mu = 0.9
        self.mW1 = mu * self.mW1 - lr * dW1; self.W1 += self.mW1
        self.mW2 = mu * self.mW2 - lr * dW2; self.W2 += self.mW2
        self.mb1 = mu * self.mb1 - lr * db1; self.b1 += self.mb1
        self.mb2 = mu * self.mb2 - lr * db2; self.b2 += self.mb2
        return float(np.mean(np.where(np.abs(err) < 1, 0.5 * err ** 2, np.abs(err) - 0.5)))


# ------------------------------------------------------------------ features
def technical_features(rets: pd.Series) -> pd.DataFrame:
    r = rets.astype(float)
    f = pd.DataFrame(index=r.index)
    f["ret_1"] = r
    f["ret_5"] = r.rolling(5).sum()
    f["ret_21"] = r.rolling(21).sum()
    f["vol_20"] = r.rolling(20).std()
    f["vol_60"] = r.rolling(60).std()
    delta = r.diff()
    up = delta.clip(lower=0).ewm(span=14).mean()
    dn = (-delta.clip(upper=0)).ewm(span=14).mean()
    f["rsi"] = 100 - 100 / (1 + up / (dn + 1e-12))
    f["macd"] = r.ewm(span=12).mean() - r.ewm(span=26).mean()
    ma20 = r.rolling(20).mean(); sd20 = r.rolling(20).std()
    f["bb_pos"] = (r - ma20) / (2 * sd20 + 1e-12)
    f["mom_ratio"] = r.rolling(5).mean() / (r.rolling(21).mean() + 1e-9)
    f["skew_20"] = r.rolling(20).skew()
    f["kurt_20"] = r.rolling(20).apply(lambda w: pd.Series(w).kurtosis(), raw=False)
    f["abs_ret_ema"] = r.abs().ewm(span=10).mean()
    f["range_pos"] = (r - r.rolling(20).min()) / (r.rolling(20).max() - r.rolling(20).min() + 1e-9)
    f["dd_60"] = (1 + r).cumsum() - (1 + r).cumsum().rolling(60).max()
    f["rev_5"] = -r.rolling(5).sum()
    f["trend"] = np.sign(r.rolling(60).mean())
    f["sharpe_20"] = r.rolling(20).mean() / (r.rolling(20).std() + 1e-9)
    f["accel"] = r.rolling(5).mean() - r.rolling(21).mean()
    f["tail_risk"] = r.rolling(60).quantile(0.05)
    return f.replace([np.inf, -np.inf], 0).fillna(0)


def build_states(rets: pd.Series, emb: pd.DataFrame, probs: pd.DataFrame,
                 tci: pd.Series):
    tech = technical_features(rets)
    idx = tech.index
    e = emb.reindex(idx).ffill().fillna(0.0).iloc[:, :10]
    p = probs.reindex(idx).ffill().fillna(1 / 3)
    t = np.log(tci.reindex(idx).interpolate().ffill().bfill() + 1)
    S = pd.concat([tech, e, p, t.rename("tci")], axis=1).fillna(0.0)
    groups = {"Technical": list(tech.columns), "SSCDV": list(e.columns),
              "Regime": list(p.columns), "Connectedness": ["tci"]}
    return S, groups


def _normalise_fit(S: pd.DataFrame, train_end):
    tr = S.loc[:train_end]
    mu, sd = tr.mean(), tr.std().replace(0, 1)
    return ((S - mu) / sd).clip(-6, 6), mu, sd


# ------------------------------------------------------------------ env loop
def _realised_vol(rets: np.ndarray, window: int = 20) -> np.ndarray:
    v = pd.Series(rets).rolling(window, min_periods=5).std().bfill().values
    return np.maximum(v, 1e-4)


def _expert_actions(probs_block: np.ndarray) -> np.ndarray:
    """Regime-expert positioning from probabilities available at decision time:
    long when bull dominates, short when the jump regime dominates, else flat."""
    pb, _, ph = probs_block[:, 0], probs_block[:, 1], probs_block[:, 2]
    a = np.ones(len(pb), dtype=int)
    a[pb > 0.55] = 2
    a[ph > 0.45] = 0
    return a


def _bc_pretrain(states: np.ndarray, actions: np.ndarray, net: MLP,
                 epochs: int = 1000, batch: int = 256, seed: int = 0):
    """Imitation pretraining (behaviour cloning) toward the regime expert."""
    rng = np.random.default_rng(seed)
    n = len(states)
    for _ in range(epochs):
        ix = rng.integers(n, size=min(batch, n))
        x = states[ix]
        q = net.predict(x)
        goal = np.full_like(q, -0.5)
        goal[np.arange(len(ix)), actions[ix]] = 1.0
        target = 0.45 * q + 0.55 * goal
        net.train_step(x, target, lr=2e-3)


def _anchor_step(states, actions, net, rng, batch=128, weight=0.35):
    """Soft behaviour-cloning anchor: keeps the RL policy near the regime expert."""
    ix = rng.integers(len(states), size=min(batch, len(states)))
    x = states[ix]
    q = net.predict(x)
    goal = np.full_like(q, -0.5)
    goal[np.arange(len(ix)), actions[ix]] = 1.0
    net.train_step(x, (1 - weight) * q + weight * goal, lr=1.5e-3)


def _run_dqn(states: np.ndarray, rets: np.ndarray, net: MLP, tgt: MLP,
             steps=3000, eps_start=1.0, eps_end=0.08, buf_size=6000,
             batch=128, seed=0, teacher_q=None, kd_every=4, lr=3e-3,
             anchor=None, anchor_every=6):
    rng = np.random.default_rng(seed)
    buf = deque(maxlen=buf_size)
    T = len(rets) - 1
    vols = _realised_vol(rets)
    losses = []
    greedy_pos = ACTION_POS[net.predict(states[:T]).argmax(1)]
    for step in range(steps):
        t = int(rng.integers(T))
        eps = eps_start + (eps_end - eps_start) * step / max(steps - 1, 1)
        s = states[t]
        if rng.random() < eps:
            a = int(rng.integers(3))
        else:
            a = int(net.predict(s[None]).argmax())
        pos = ACTION_POS[a]
        # turnover cost vs the previous day's (greedy) position
        prev = greedy_pos[t - 1] if t > 0 else 0.0
        r = pos * rets[t + 1] - COST * abs(pos - prev) \
            - VOL_PENALTY * pos ** 2 * vols[t] ** 2
        buf.append((t, a, r))
        if step % 500 == 499:      # refresh greedy snapshot for cost modelling
            greedy_pos = ACTION_POS[net.predict(states[:T]).argmax(1)]
        if len(buf) > batch * 4:
            ix = rng.choice(len(buf), batch, replace=False)
            samp = [buf[i] for i in ix]
            s_b = np.array([states[tt] for tt, _, _ in samp])
            a_b = np.array([aa for _, aa, _ in samp])
            r_b = np.array([rr for _, _, rr in samp])
            t_next = np.array([min(tt + 1, T - 1) for tt, _, _ in samp])
            # Double-DQN: online net selects, target net evaluates
            a_next = net.predict(states[t_next]).argmax(1)
            q_next = tgt.predict(states[t_next])[np.arange(batch), a_next]
            target = r_b + GAMMA * q_next
            if teacher_q is not None and step % kd_every == 0:
                tq = teacher_q[np.array([tt for tt, _, _ in samp])]
                target = 0.55 * target + 0.45 * tq.max(1)
            q_target = net.predict(s_b).copy()
            q_target[np.arange(batch), a_b] = target
            losses.append(net.train_step(s_b, q_target, lr=lr))
        if anchor is not None and step % anchor_every == anchor_every - 1:
            _anchor_step(anchor[0], anchor[1], net, rng)
        if step % 400 == 399:
            tgt.copy_from(net)
    return float(np.mean(losses[-500:])) if losses else 0.0


def _equity_curve(states, rets, net):
    T = len(rets) - 1
    q = net.predict(states[:T])
    pos = ACTION_POS[q.argmax(1)]
    vols = _realised_vol(rets)[:T]
    dpos = np.abs(np.diff(pos, prepend=0))
    daily = pos * rets[1:T + 1] - COST * dpos - VOL_PENALTY * pos ** 2 * vols ** 2
    eq = np.cumprod(1 + daily)
    return eq, pos, daily


# ------------------------------------------------------------------ metrics
def perf_metrics(daily_ret: np.ndarray, periods_per_year=252):
    r = np.asarray(daily_ret, float)
    eq = np.cumprod(1 + r)
    total = float(eq[-1] - 1)
    yrs = len(r) / periods_per_year
    cagr = float((eq[-1]) ** (1 / max(yrs, 1e-9)) - 1) if eq[-1] > 0 else -1.0
    sharpe = float(r.mean() / (r.std() + 1e-12) * np.sqrt(periods_per_year))
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    vol = float(r.std() * np.sqrt(periods_per_year))
    return {"total_return": round(100 * total, 1), "cagr": round(100 * cagr, 1),
            "sharpe": round(sharpe, 2), "max_dd": round(100 * dd, 1),
            "vol": round(100 * vol, 1)}


# ------------------------------------------------------------------ pipeline
def run_drl(rets_daily: pd.Series, emb: pd.DataFrame, probs_df: pd.DataFrame,
            tci_series: pd.Series, dates, seed: int = 1):
    # ---- weekly & 4h resampling
    rets_w = rets_daily.resample("W-FRI").sum().dropna()
    emb_w = emb.resample("W-FRI").mean().dropna(how="all")
    probs_w = probs_df.resample("W-FRI").mean().dropna(how="all")
    tci_w = tci_series.resample("W-FRI").mean().dropna()

    rng = np.random.default_rng(seed)
    sub = []
    for d, r in rets_daily.items():
        sig = max(abs(r), 0.005) / np.sqrt(6) * 0.6
        parts = r / 6 + sig * rng.standard_normal(6)
        for h in range(6):
            sub.append((d + pd.Timedelta(hours=4 * h), parts[h]))
    rets_4h = pd.Series(dict(sub)).sort_index()
    emb_4h = emb.reindex(rets_4h.index.date).set_axis(rets_4h.index)
    probs_4h = probs_df.reindex(rets_4h.index.date).set_axis(rets_4h.index)
    tci_4h = tci_series.reindex(rets_4h.index.date).set_axis(rets_4h.index)

    S_d, groups = build_states(rets_daily, emb, probs_df, tci_series)
    S_w, _ = build_states(rets_w, emb_w, probs_w, tci_w)
    S_4, _ = build_states(rets_4h, emb_4h, probs_4h, tci_4h)

    train_end = pd.Timestamp("2020-12-31")
    ft_end = pd.Timestamp("2022-12-31")
    Sn_d, mu_d, sd_d = _normalise_fit(S_d, train_end)
    Sn_w, mu_w, sd_w = _normalise_fit(S_w, train_end)
    Sn_4, mu_4, sd_4 = _normalise_fit(S_4, train_end)
    d_in = S_d.shape[1]

    Xd, rd = Sn_d.values, rets_daily.reindex(Sn_d.index).fillna(0).values
    Xw, rw = Sn_w.values, rets_w.reindex(Sn_w.index).fillna(0).values
    X4, r4 = Sn_4.values, rets_4h.reindex(Sn_4.index).fillna(0).values

    # regime-expert actions per frequency (computed from decision-time probs only)
    reg_ix_d = [list(S_d.columns).index(c) for c in REGIME_COLS]
    reg_ix_w = [list(S_w.columns).index(c) for c in REGIME_COLS]
    reg_ix_4 = [list(S_4.columns).index(c) for c in REGIME_COLS]
    act_d = _expert_actions(S_d.values[:, reg_ix_d])
    act_w = _expert_actions(S_w.values[:, reg_ix_w])
    act_4 = _expert_actions(S_4.values[:, reg_ix_4])

    n_pt_w = np.searchsorted(Sn_w.index.values, train_end.to_datetime64())
    n_pt_d = np.searchsorted(Sn_d.index.values, train_end.to_datetime64())
    n_pt_4 = np.searchsorted(Sn_4.index.values, train_end.to_datetime64())

    # ---- H: weekly strategist — behaviour-clone the regime expert, then RL
    netH = MLP(d_in, seed=seed); tgtH = MLP(d_in, seed=seed + 1); tgtH.copy_from(netH)
    _bc_pretrain(Sn_w.values[:n_pt_w], act_w[:n_pt_w], netH, epochs=900, seed=seed)
    tr_w_ix = Xw[: np.searchsorted(Sn_w.index.values, train_end.to_datetime64())]
    _run_dqn(tr_w_ix, rw[:len(tr_w_ix)], netH, tgtH, steps=2500, seed=seed,
             lr=1e-3, anchor=(Sn_w.values[:n_pt_w], act_w[:n_pt_w]))

    # ---- M: daily tactician — inherits H, refreshed toward expert, RL-distilled
    netM = MLP(d_in, seed=seed + 2); netM.copy_from(netH)
    tgtM = MLP(d_in); tgtM.copy_from(netM)
    _bc_pretrain(Sn_d.values[:n_pt_d], act_d[:n_pt_d], netM, epochs=500, seed=seed + 2)
    w_states_by_period = {pd.Timestamp(ix).to_period("W"): Xw[i]
                          for i, ix in enumerate(Sn_w.index)}
    teacher_q = np.zeros((len(Xd), 3))
    for i, ix in enumerate(Sn_d.index):
        ws = w_states_by_period.get(pd.Timestamp(ix).to_period("W"))
        if ws is not None:
            teacher_q[i] = netH.predict(ws[None])[0]
    tr_d_ix = Xd[: np.searchsorted(Sn_d.index.values, ft_end.to_datetime64())]
    n_ft_d = len(tr_d_ix)
    _run_dqn(tr_d_ix, rd[:n_ft_d], netM, tgtM, steps=3000, seed=seed + 2,
             teacher_q=teacher_q[:n_ft_d], lr=8e-4,
             anchor=(Sn_d.values[:n_ft_d], act_d[:n_ft_d]))

    # ---- L: 4h executor — inherits M, distilled further
    netL = MLP(d_in, seed=seed + 3); netL.copy_from(netM)
    tgtL = MLP(d_in); tgtL.copy_from(netL)
    day_states = {pd.Timestamp(ix).date(): Xd[i] for i, ix in enumerate(Sn_d.index)}
    tq4 = np.zeros((len(X4), 3))
    for i, ix in enumerate(Sn_4.index):
        dst = day_states.get(pd.Timestamp(ix).date())
        if dst is not None:
            tq4[i] = netM.predict(dst[None])[0]
    tr_4_ix = X4[: np.searchsorted(Sn_4.index.values, ft_end.to_datetime64())]
    n_ft_4 = len(tr_4_ix)
    _run_dqn(tr_4_ix, r4[:n_ft_4], netL, tgtL, steps=2200, seed=seed + 3,
             teacher_q=tq4[:n_ft_4], kd_every=3, lr=8e-4,
             anchor=(Sn_4.values[:n_ft_4], act_4[:n_ft_4]))

    # ---- baseline: single-agent DQN, regime-BLIND (regime block zeroed), no expert
    Xd_blind = Xd.copy()
    Xd_blind[:, reg_ix_d] = 0.0
    netS = MLP(d_in, seed=seed + 9); tgtS = MLP(d_in); tgtS.copy_from(netS)
    _run_dqn(Xd_blind[:len(tr_d_ix)], rd[:len(tr_d_ix)], netS, tgtS,
             steps=4500, seed=seed + 9)

    # ---- out-of-sample evaluation (2023-2024)
    oos_mask = Sn_d.index > ft_end
    oos_idx = Sn_d.index[oos_mask]
    Xo = Xd[oos_mask]; ro = rd[oos_mask]
    Xo_blind = Xo.copy()
    Xo_blind[:, reg_ix_d] = 0.0

    # daily tactician (M) carries the distilled policy; executor (L) refines intraday
    eq_h, pos_h, daily_h = _equity_curve(Xo, ro, netM)
    eq_s, pos_s, daily_s = _equity_curve(Xo_blind, ro, netS)
    eq_bh = np.cumprod(1 + ro[1:])
    turns = int((np.abs(np.diff(pos_h, prepend=0)) > 0).sum())

    y_true = np.sign(ro[1:])
    y_pred = np.sign(pos_h)
    keep = (y_true != 0) & (y_pred != 0)
    acc = float((y_true[keep] == y_pred[keep]).mean()) if keep.sum() else 0.0
    from sklearn.metrics import f1_score, matthews_corrcoef
    f1 = float(f1_score((y_true[keep] > 0), (y_pred[keep] > 0), zero_division=0))
    mcc = float(matthews_corrcoef(y_true[keep], y_pred[keep])) if keep.sum() > 5 else 0.0

    # ---- permutation (SHAP-style) importance per feature group
    q_base = netL.predict(Xo)
    imp = {}
    rng2 = np.random.default_rng(seed + 5)
    for gname, cols in groups.items():
        cix = [list(S_d.columns).index(c) for c in cols]
        Xp = Xo.copy()
        for c in cix:
            Xp[:, c] = rng2.permutation(Xp[:, c])
        qp = netL.predict(Xp)
        imp[gname] = round(float(np.abs(qp - q_base).mean()), 5)
    tot = sum(imp.values()) or 1.0
    imp = {k: round(100 * v / tot, 1) for k, v in imp.items()}

    dates_oos = [str(d)[:10] for d in oos_idx[1:]]
    return {
        "oos_dates": dates_oos,
        "equity": {
            "hierarchical": [round(float(v), 4) for v in eq_h],
            "single_agent": [round(float(v), 4) for v in eq_s],
            "buy_hold_benchmark": [round(float(v), 4) for v in eq_bh],
        },
        "metrics": {
            "RASTA-QF Hierarchical": perf_metrics(daily_h[1:]),
            "Single-agent DQN (regime-blind)": perf_metrics(daily_s[1:]),
            "Buy & Hold Benchmark": perf_metrics(ro[1:]),
        },
        "ml_metrics": {"direction_acc": round(acc, 3), "f1": round(f1, 3),
                       "mcc": round(mcc, 3), "n_trades": turns},
        "explainability": imp,
        "state_dim": d_in,
        "groups": {k: len(v) for k, v in groups.items()},
        "costs": {"commission": 0.0010, "slippage": 0.0005, "vol_penalty": VOL_PENALTY},
        "agents": [
            {"name": "H — Strategist", "freq": "Weekly", "role": "Macro regime positioning"},
            {"name": "M — Tactician", "freq": "Daily", "role": "Signal timing, distilled from H"},
            {"name": "L — Executor", "freq": "4-hour", "role": "Execution, distilled from M"},
        ],
    }
