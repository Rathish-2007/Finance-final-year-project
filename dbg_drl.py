import os, sys
sys.path.insert(0, '.')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rastaqf.settings")
import django; django.setup()
import numpy as np, pandas as pd
from core.services import data_lake, cjm, drl, preprocess, sscdv, tvpvar

ds = data_lake.build_dataset(seed=42)
port = ds["returns"][["SPX", "GRNB", "GOLD", "OIL", "CLEAN"]].mean(axis=1)
fit = cjm.fit_cjm(port.values, n_states=3, iters=80, n_starts=8)
probs = pd.DataFrame(fit["probs"], index=port.index, columns=["p_bull", "p_bear", "p_high"])
corpus = sscdv.generate_corpus(ds["prices"].index, ds["states"], n_docs=6000)
sout = sscdv.build_sscdv(corpus, n_components=50)
z = preprocess.standardise(ds["returns"][data_lake.CORE7])
qqc = tvpvar.rolling_qqc(z.values, z.index, window=252, step=21)
tci_med = pd.Series(qqc["series"][0.5]["tci"], index=pd.DatetimeIndex(qqc["series"][0.5]["dates"]))
tci_daily = tci_med.reindex(ds["prices"].index, method="ffill").bfill()

bench = ds["returns"][data_lake.CORE7].mean(axis=1)

# replicate run_drl internals but expose each net
import importlib
res = drl.run_drl(bench, sout["daily_emb"], probs, tci_daily, ds["prices"].index)
print("hierarchical:", res["metrics"]["RASTA-QF Hierarchical"])
print("single      :", res["metrics"]["Single-agent DQN (regime-blind)"])
print("buyhold     :", res["metrics"]["Buy & Hold Benchmark"])
print("ml_metrics  :", res["ml_metrics"])
