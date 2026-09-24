import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/kawai/sector-rotation")
from build_dashboard import causal_gaussian, cooldown
sys.stdout.reconfigure(encoding="utf-8")
d = pd.read_parquet(r"C:/Users/kawai/sector-rotation/research/t17.parquet"); d = d[d.index >= "2009-01-05"].ffill(limit=3)
r = d.pct_change(); r = r.mask((r > 0.3) | (r < -0.4), 0.0)
topix = r["1306.T"]; etf = r.drop(columns="1306.T")
lr = np.log1p(etf); rel = lr.sub(lr.mean(axis=1), axis=0) * 100
flow = causal_gaussian(rel.fillna(0), 8)
off = ["1617.T", "1621.T", "1627.T", "1628.T"]
on = [c for c in flow.columns if c not in off]
rot = flow[on].mean(axis=1) - flow[off].mean(axis=1)
sig = rot.rolling(120, min_periods=60).std()
mb = cooldown((rot < -sig) & (rot.shift() >= -sig.shift()))
pb = cooldown((rot > sig) & (rot.shift() <= sig.shift()))
tp = (1 + topix.fillna(0)).cumprod()
def fwd(h): return (tp.shift(-(h + 1)) / tp.shift(-1) - 1) * 100   # 翌日終値エントリー
start = "2009-07-01"
for name, ev in [("−σブレイク", mb), ("+σブレイク", pb)]:
    ev = ev[start:]; ev = ev[ev].index
    print(f"\n{name}: {len(ev)}回")
    for h in [20, 60, 120]:
        f = fwd(h)[start:].dropna(); e = f.reindex(ev).dropna()
        # 無条件平均との差を、ランダム日付抽出で検定
        rng = np.random.default_rng(0)
        sims = [f.sample(len(e), random_state=int(rng.integers(1e9))).mean() for _ in range(5000)]
        p = (np.array(sims) >= e.mean()).mean()
        print(f"  {h:>3}日後 TOPIX: 平均{e.mean():+.2f}% 勝率{(e>0).mean()*100:.0f}% | 無条件{f.mean():+.2f}% | 上回る確率(偶然)p={p:.3f}")
    for per in [("2009", "2015"), ("2016", "2020"), ("2021", "2026")]:
        f = fwd(60)[per[0]:per[1]]; e = f.reindex(ev[(ev >= per[0]) & (ev <= per[1] + "-12-31")]).dropna()
        print(f"   {per[0]}-{per[1]} 60日: 事象{len(e)}回 平均{e.mean():+.2f}% vs 無条件{f.mean():+.2f}%")
