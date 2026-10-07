import os, sys
sys.path.insert(0, '.')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rastaqf.settings")
import django; django.setup()
import numpy as np, pandas as pd
from core.services import data_lake, cjm, drl

ds = data_lake.build_dataset(seed=42)
truth = ds["states"][1:]
port = ds["returns"][["SPX", "GRNB", "GOLD", "OIL", "CLEAN"]].mean(axis=1)
core7 = ds["returns"][data_lake.CORE7].mean(axis=1)
fit = cjm.fit_cjm(port.values, n_states=3, iters=80, n_starts=8)
probs = pd.DataFrame(fit["probs"], index=port.index, columns=["p_bull", "p_bear", "p_high"])

print("== CORE7 equal-weight portfolio yearly ==")
for yr in range(2016, 2025):
    m = [d.year == yr for d in core7.index]
    r = core7.values[m]
    print(f"{yr}: ret={(np.exp(r.sum())-1)*100:7.1f}%  vol={r.std()*100:4.1f}%")

oos = core7.index > pd.Timestamp("2022-12-31")
p_oos = probs.reindex(core7.index[oos]).ffill().fillna(1/3).values
r_oos = core7.values[oos]
print("\nOOS regime dominance:", np.bincount(p_oos.argmax(1), minlength=3))
dom = p_oos.argmax(1)
for k in range(3):
    sel = dom[:-1] == k
    print(f"dom {k}: mean CORE7 ret {r_oos[1:][sel].mean()*100:+.3f}%/day  n={sel.sum()}")

a = drl._expert_actions(p_oos)
pos = drl.ACTION_POS[a]
daily = pos * r_oos - 0.0015 * np.abs(np.diff(pos, prepend=0))
eq = np.cumprod(1 + daily[1:])
print("\nEXPERT OOS eq: %.3f | CORE7 buyhold %.3f" % (eq[-1], np.cumprod(1 + r_oos[1:])[-1]))
print("expert daily mean %.4f std %.4f sharpe %.2f" % (
    daily[1:].mean(), daily[1:].std(), daily[1:].mean()/daily[1:].std()*np.sqrt(252)))
