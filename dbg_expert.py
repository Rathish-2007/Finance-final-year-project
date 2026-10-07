import os, sys
sys.path.insert(0, '.')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rastaqf.settings")
import django; django.setup()
import numpy as np, pandas as pd
from core.services import data_lake, cjm, drl

ds = data_lake.build_dataset(seed=42)
btc = ds["returns"]["BTC"]
port = ds["returns"][["SPX", "GRNB", "GOLD", "OIL", "CLEAN"]].mean(axis=1)
fit = cjm.fit_cjm(port.values, n_states=3, iters=80, n_starts=8)
probs = pd.DataFrame(fit["probs"], index=port.index, columns=["p_bull", "p_bear", "p_high"])
oos = btc.index > pd.Timestamp("2022-12-31")
p_oos = probs.reindex(btc.index[oos]).ffill().fillna(1 / 3).values
r_oos = btc.values[oos]
print("OOS regime dominance counts:", np.bincount(p_oos.argmax(1), minlength=3))
a = drl._expert_actions(p_oos)
pos = drl.ACTION_POS[a]
daily = pos * r_oos - 0.0015 * np.abs(np.diff(pos, prepend=0))
eq = np.cumprod(1 + daily[1:])
print("EXPERT OOS eq: %.2f | buyhold %.3f" % (eq[-1], np.cumprod(1 + r_oos[1:])[-1]))
print("expert pos dist:", dict(zip(*np.unique(pos, return_counts=True))))
print("expert daily mean %.4f std %.4f sharpe %.2f" % (
    daily[1:].mean(), daily[1:].std(), daily[1:].mean() / daily[1:].std() * np.sqrt(252)))
dom = p_oos.argmax(1)
for k in range(3):
    sel = dom[:-1] == k
    print(f"dom {k}: mean BTC ret {r_oos[1:][sel].mean()*100:+.3f}%/day  n={sel.sum()}")
