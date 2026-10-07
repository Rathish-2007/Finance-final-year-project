"""Quick smoke test of every analytics module with reduced sizes."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rastaqf.settings")
import django; django.setup()

import numpy as np, pandas as pd
from core.services import data_lake, preprocess, sscdv, cjm, tvpvar, risk, drl

t0 = time.time()
ds = data_lake.build_dataset(seed=42)
print(f"[data] prices {ds['prices'].shape} in {time.time()-t0:.1f}s")

t0 = time.time()
corpus = sscdv.generate_corpus(ds["prices"].index, ds["states"], n_docs=2500)
out = sscdv.build_sscdv(corpus, n_components=50)
print(f"[sscdv] emb {out['daily_emb'].shape} var={out['pca_var']:.2f} in {time.time()-t0:.1f}s")

t0 = time.time()
z = preprocess.standardise(ds["returns"][data_lake.CORE7])
port = ds["returns"][["SPX", "GRNB", "GOLD", "OIL", "CLEAN"]].mean(axis=1)
fit = cjm.fit_cjm(port.values, n_states=3, iters=40)
acc = (fit["path"] == ds["states"][1:]).mean()
print(f"[cjm] acc={acc:.3f} mus={np.round(fit['mus'],2)} sigs={np.round(fit['sigs'],2)} in {time.time()-t0:.1f}s")

t0 = time.time()
qqc = tvpvar.rolling_qqc(z.values[-600:], z.index[-600:], window=252, step=63)
print(f"[qqc] tci med tail={qqc['series'][0.5]['tci'][-3:]} in {time.time()-t0:.1f}s")

t0 = time.time()
r_aux = ds["returns"].reindex(ds["aux"].index).fillna(0)
h = risk.hansen_threshold((r_aux["OIL"]+r_aux["GOLD"]).values,
                          ds["aux"]["ENERGY_DEP"].values, r_aux["SPX"].values, n_boot=80)
sc = risk.stablecoin_run_risk(ds["aux"]["PEG_DEV"].values, ds["aux"]["BTC_RET"].values,
                              ds["aux"]["FUNDING"].values)
print(f"[risk] hansen F={h['F_stat']} p={h['p_value']} thr={h['threshold']} scTCI={sc['tci']} in {time.time()-t0:.1f}s")

t0 = time.time()
probs = pd.DataFrame(fit["probs"], index=port.index, columns=["p_bull","p_bear","p_high"])
tci = pd.Series(qqc["series"][0.5]["tci"], index=pd.DatetimeIndex(qqc["series"][0.5]["dates"]))
tci_daily = tci.reindex(ds["prices"].index, method="ffill").bfill()
r = drl.run_drl(ds["returns"][data_lake.CORE7].mean(axis=1), out["daily_emb"], probs, tci_daily, ds["prices"].index)
print(f"[drl] metrics={r['metrics']['RASTA-QF Hierarchical']} imp={r['explainability']} in {time.time()-t0:.1f}s")
print("SMOKE OK")
