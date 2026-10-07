"""Pipeline orchestrator — runs the 6 RASTA-QF modules and persists results."""
from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
from django.conf import settings
from django.utils import timezone

from ..models import ModuleState, RunLog
from . import data_lake, preprocess, sscdv, cjm, tvpvar, risk, drl

MODULE_ORDER = ["data", "sscdv", "regimes", "connectedness", "risk", "drl"]
MEMO = {}          # in-memory heavy objects shared across modules
_LOCK = threading.Lock()


# ------------------------------------------------------------------ helpers
def _state(name: str) -> ModuleState:
    st, _ = ModuleState.objects.get_or_create(name=name)
    return st


def _save(name, **kw):
    st = _state(name)
    for k, v in kw.items():
        setattr(st, k, v)
    st.save()


def _log(module, msg, level="INFO"):
    RunLog.objects.create(module=module, level=level, message=msg)


def decimate(s, max_pts=600):
    a = list(s)
    if len(a) > max_pts:
        k = int(np.ceil(len(a) / max_pts))
        a = a[::k]
    return a


def to_series(series: pd.Series, max_pts=600, date_fmt="%Y-%m-%d"):
    s = series.dropna()
    k = max(1, int(np.ceil(len(s) / max_pts)))
    s = s.iloc[::k]
    return {"dates": [d.strftime(date_fmt) for d in s.index],
            "values": [round(float(v), 6) for v in s.values]}


def jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, np.ndarray):
        return [jsonable(v) for v in x]
    return x


# ------------------------------------------------------------------ module 1
def run_data(cb):
    _log("data", f"Building unified 10-asset data lake (seed={settings.DATA_SEED})")
    ds = data_lake.build_dataset(seed=settings.DATA_SEED,
                                 use_yfinance=settings.USE_YFINANCE)
    MEMO["ds"] = ds
    prices, returns, aux = ds["prices"], ds["returns"], ds["aux"]
    cb(0.5)
    stats = []
    for a in ds["asset_meta"]:
        s = a["sym"]
        r = returns[s]
        stats.append({
            "sym": s, "name": a["name"], "cls": a["cls"], "ticker": a["ticker"],
            "ann_ret": round(float(r.mean() * 252 * 100), 1),
            "ann_vol": round(float(r.std() * np.sqrt(252) * 100), 1),
            "sharpe": round(float(r.mean() / (r.std() + 1e-12) * np.sqrt(252)), 2),
            "max_dd": round(float(((prices[s] / prices[s].cummax()) - 1).min() * 100), 1),
            "skew": round(float(r.skew()), 2),
            "kurt": round(float(r.kurtosis()), 2),
        })
    corr = returns[data_lake.CORE7].corr()
    results = {
        "source": ds["source"],
        "n_days": len(prices),
        "start": str(prices.index[0].date()), "end": str(prices.index[-1].date()),
        "stats": stats,
        "corr": [[round(float(corr.iloc[i, j]), 3) for j in range(len(corr))]
                 for i in range(len(corr))],
        "corr_labels": list(corr.columns),
        "prices": {s: to_series(prices[s], 450) for s in ["SPX", "BTC", "GOLD", "GRNB", "OIL"]},
        "svi": to_series(aux["SVI"], 450),
        "news_count": to_series(aux["NEWS_COUNT"].astype(float), 450),
    }
    cb(1.0)
    _log("data", f"Data lake ready: {len(prices)} trading days × {prices.shape[1]} assets")
    return results


# ------------------------------------------------------------------ module 2
def run_sscdv(cb):
    ds = _need("data")
    cb(0.2)
    corpus = sscdv.generate_corpus(ds["prices"].index, ds["states"], n_docs=6000)
    cb(0.45)
    out = sscdv.build_sscdv(corpus, n_components=50)
    MEMO["sscdv"] = out
    cb(0.85)
    results = {
        "n_docs": out["n_docs"], "vocab_size": out["vocab_size"],
        "pca_var": round(100 * out["pca_var"], 1),
        "topic_totals": {k: int(v) for k, v in out["topic_totals"].items()},
        "sentiment": to_series(out["daily_sent"], 500),
        "news_volume": to_series(out["daily_news"].astype(float), 500),
        "topic_mix": {t: to_series(out["daily_topic"][t], 400)
                      for t in sscdv.TOPICS},
    }
    cb(1.0)
    _log("sscdv", f"SSCDV built: {out['n_docs']} docs → 50-dim embeddings "
                  f"(PCA var {results['pca_var']}%)")
    return results


# ------------------------------------------------------------------ module 3
def run_regimes(cb):
    ds = _need("data")
    # CJM estimates macro regimes from the traditional-asset portfolio;
    # crypto then inherits the regime probabilities via the state space.
    port = ds["returns"][["SPX", "GRNB", "GOLD", "OIL", "CLEAN"]].mean(axis=1)
    cb(0.3)
    fit = cjm.fit_cjm(port.values, n_states=3, iters=25)
    MEMO["regimes"] = fit
    idx = port.index
    probs = pd.DataFrame(fit["probs"], index=idx,
                         columns=["p_bull", "p_bear", "p_high"])
    MEMO["probs_df"] = probs
    acc = float((fit["path"] == ds["states"][1:]).mean())
    cb(0.8)
    results = {
        "labels": fit["labels"],
        "probs": {c: to_series(probs[c], 500) for c in probs.columns},
        "A": [[round(float(v), 4) for v in row] for row in fit["A"]],
        "mus": [round(float(v), 4) for v in fit["mus"]],
        "sigs": [round(float(v), 4) for v in fit["sigs"]],
        "bic": round(fit["bic"], 1), "loglik": round(fit["loglik"], 1),
        "state_accuracy_vs_truth": round(100 * acc, 1),
        "params": fit["params"],
        "port_ret": to_series(port, 500),
    }
    cb(1.0)
    _log("regimes", f"CJM fitted: BIC={results['bic']}, "
                    f"state recovery vs synthetic truth {results['state_accuracy_vs_truth']}%")
    return results


# ------------------------------------------------------------------ module 4
def run_connectedness(cb):
    ds = _need("data")
    z = preprocess.standardise(ds["returns"][data_lake.CORE7])
    idx = z.index

    def pcb(f):
        cb(0.05 + 0.9 * f)

    out = tvpvar.rolling_qqc(z.values, idx, window=252, step=21, progress_cb=pcb)
    MEMO["qqc"] = out
    # daily-frequency TCI interpolation (median quantile) for the DRL state
    tci_med = pd.Series(out["series"][0.5]["tci"],
                        index=pd.DatetimeIndex(out["series"][0.5]["dates"]))
    tci_daily = tci_med.reindex(idx, method="ffill").bfill()
    MEMO["tci_daily"] = tci_daily
    results = {
        "assets": data_lake.CORE7,
        "window": out["window"], "horizon": out["horizon"],
        "series": {str(t): {"dates": out["series"][t]["dates"],
                            "tci": out["series"][t]["tci"]} for t in (0.05, 0.5, 0.95)},
        "summary": out["summary"],
        "tci_daily": to_series(tci_daily, 500),
    }
    cb(1.0)
    _log("connectedness", f"QQC computed over {len(out['series'][0.5]['dates'])} windows × 3 quantiles")
    return results


# ------------------------------------------------------------------ module 5
def run_risk(cb):
    ds = _need("data")
    rets, aux = ds["returns"], ds["aux"]
    cb(0.1)
    y = (rets["OIL"] + rets["GOLD"]).reindex(aux.index).fillna(0).values
    h = risk.hansen_threshold(y, aux["ENERGY_DEP"].values, rets["SPX"].reindex(aux.index).fillna(0).values,
                              n_boot=250)
    cb(0.55)
    g = pd.DataFrame({
        "FINTECH": np.log(aux["FINTECH"]).diff().fillna(0),
        "BANK_STAB": np.log(aux["BANK_STAB"]).diff().fillna(0),
        "SVI": aux["SVI"].diff().fillna(0),
        "BTC_RET": aux["BTC_RET"].fillna(0),
        "PEG_DEV": aux["PEG_DEV"].diff().fillna(0),
    })
    pairs = [("FINTECH", "BANK_STAB"), ("BANK_STAB", "FINTECH"),
             ("SVI", "BTC_RET"), ("BTC_RET", "SVI"),
             ("PEG_DEV", "BTC_RET")]
    battery = risk.granger_battery(g, pairs, maxlag=5)
    cb(0.8)
    sc = risk.stablecoin_run_risk(aux["PEG_DEV"].values, aux["BTC_RET"].values,
                                  aux["FUNDING"].values)
    cb(1.0)
    _log("risk", f"Hansen sup-F={h['F_stat']} (p={h['p_value']}), "
                 f"stablecoin TCI={sc['tci']}")
    return {"hansen": h, "granger": battery, "stablecoin": sc}


# ------------------------------------------------------------------ module 6
def run_drl(cb):
    ds = _need("data")
    s = _need("sscdv", "M2 SSCDV")
    p = _need("regimes", "M3 Regimes")
    q = _need("connectedness", "M4 Connectedness")
    cb(0.1)
    out = drl.run_drl(
        rets_daily=ds["returns"][data_lake.CORE7].mean(axis=1),   # unified benchmark index
        emb=s["daily_emb"],
        probs_df=MEMO["probs_df"],
        tci_series=MEMO["tci_daily"],
        dates=ds["prices"].index,
    )
    cb(1.0)
    _log("drl", f"DRL trained: state dim {out['state_dim']}, "
                f"OOS Sharpe {out['metrics']['RASTA-QF Hierarchical']['sharpe']}")
    return out


# ------------------------------------------------------------------ runner
RUNNERS = {"data": run_data, "sscdv": run_sscdv, "regimes": run_regimes,
           "connectedness": run_connectedness, "risk": run_risk, "drl": run_drl}


def _need(name, label=None):
    """Return the in-memory artefact for a module, executing it if needed."""
    if name == "data":
        if "ds" not in MEMO:
            _execute("data")
        return MEMO["ds"]
    if name not in MEMO:
        _execute(name)
    return MEMO[name]


def _execute(name):
    st = _state(name)
    st.status = "running"; st.progress = 0; st.message = "running…"
    st.started_at = timezone.now(); st.save()
    t0 = time.time()

    def cb(f):
        _save(name, progress=round(100 * min(f, 1.0), 1))

    try:
        results = RUNNERS[name](cb)
        _save(name, status="done", progress=100,
              message=f"completed in {time.time() - t0:.1f}s", results=results)
        MEMO.setdefault(name, results)   # runner-set rich artefacts take precedence
    except Exception as e:
        _save(name, status="error", message=f"{type(e).__name__}: {e}")
        _log(name, f"FAILED: {e}", level="ERROR")
        raise


def run_pipeline(modules=None, single_thread=False):
    mods = modules or MODULE_ORDER

    def job():
        for m in mods:
            try:
                _execute(m)
            except Exception:
                break

    if single_thread:
        job()
    else:
        with _LOCK:
            t = threading.Thread(target=job, daemon=True)
            t.start()
    return mods


def build_benchmark_file():
    """Export RASTA-QF benchmark as HDF5 (or .npz fallback)."""
    ds = _need("data")
    settings.RESULTS_DIR.mkdir(exist_ok=True)
    path = settings.RESULTS_DIR / "RASTA_QF_Benchmark.h5"
    try:
        import h5py
        with h5py.File(path, "w") as f:
            f.create_dataset("prices", data=ds["prices"].values)
            f.create_dataset("returns", data=ds["returns"].values)
            f.create_dataset("aux", data=ds["aux"].values)
            f.attrs["columns_prices"] = list(ds["prices"].columns)
            f.attrs["columns_returns"] = list(ds["returns"].columns)
            f.attrs["columns_aux"] = list(ds["aux"].columns)
            f.attrs["index"] = [str(d.date()) for d in ds["prices"].index]
            f.attrs["seed"] = settings.DATA_SEED
            f.attrs["framework"] = "RASTA-QF v1.0"
        return path, "application/x-hdf5"
    except Exception:
        path = settings.RESULTS_DIR / "RASTA_QF_Benchmark.npz"
        np.savez_compressed(path, prices=ds["prices"].values,
                            returns=ds["returns"].values, aux=ds["aux"].values)
        return path, "application/octet-stream"
