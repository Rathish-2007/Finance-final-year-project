import os, sys
sys.path.insert(0, '.')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rastaqf.settings")
import django; django.setup()
import numpy as np, pandas as pd
from core.services import data_lake, cjm, drl

ds = data_lake.build_dataset(seed=42)
port = ds["returns"][["SPX", "GRNB", "GOLD", "OIL", "CLEAN"]].mean(axis=1)
fit = cjm.fit_cjm(port.values, n_states=3, iters=80, n_starts=8)
probs = pd.DataFrame(fit["probs"], index=port.index, columns=["p_bull", "p_bear", "p_high"])
bench = ds["returns"][data_lake.CORE7].mean(axis=1)

emb0 = pd.DataFrame(np.zeros((len(bench.index), 10)), index=bench.index,
                    columns=[f"SSC{i}" for i in range(10)])
S, groups = drl.build_states(bench, emb0, probs, pd.Series(30.0, index=bench.index))
Sn, mu, sd = drl._normalise_fit(S, pd.Timestamp("2020-12-31"))
Xd = Sn.values; rd = bench.reindex(Sn.index).fillna(0).values
reg_ix = [list(S.columns).index(c) for c in drl.REGIME_COLS]
act = drl._expert_actions(S.values[:, reg_ix])

n_pt = np.searchsorted(Sn.index.values, np.datetime64("2020-12-31"))
net = drl.MLP(S.shape[1], seed=2)
drl._bc_pretrain(Sn.values[:n_pt], act[:n_pt], net, epochs=1500, seed=2)

oos = Sn.index > pd.Timestamp("2022-12-31")
Xo, ro = Xd[oos], rd[oos]
q = net.predict(Xo)
pos = drl.ACTION_POS[q.argmax(1)]
exp = drl._expert_actions(S.values[oos][:, reg_ix])
print("agreement with expert OOS:", (pos == drl.ACTION_POS[exp]).mean().round(3))
eq, _, daily = drl._equity_curve(Xo, ro, net)
print("BC-only OOS eq: %.3f | buyhold %.3f" % (eq[-1], np.cumprod(1 + ro[1:])[-1]))
print("BC-only metrics:", drl.perf_metrics(daily[1:]))
