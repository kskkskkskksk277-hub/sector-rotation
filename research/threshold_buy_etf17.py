# -*- coding: utf-8 -*-
"""threshold_buy.py の長期版。TOPIX-17業種ETFで同じ指数を作り、
ダッシュボードの-0.5と同じ「珍しさ（下位何%か）」のしきい値でTOPIXを買った場合（2009〜）"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/kawai/sector-rotation")
sys.stdout.reconfigure(encoding="utf-8")
from build_dashboard import compute, causal_gaussian
frames, series, _ = compute()
q = (series["rot"] < -0.5).mean()
d = pd.read_parquet(r"C:/Users/kawai/sector-rotation/research/t17.parquet")
d = d[d.index >= "2009-01-05"].ffill(limit=3)
r = d.pct_change(); r = r.mask((r > 0.3) | (r < -0.4), 0.0)
px = (1 + r["1306.T"].fillna(0)).cumprod(); etf = r.drop(columns="1306.T")
lr = np.log1p(etf); rel = lr.sub(lr.mean(axis=1), axis=0) * 100
flow = causal_gaussian(rel.fillna(0), 8)
off = ["1617.T", "1621.T", "1627.T", "1628.T"]
rot = (flow[[c for c in flow.columns if c not in off]].mean(axis=1) - flow[off].mean(axis=1))["2009-07":]
th = rot.quantile(q)
print(f"ダッシュボードで-0.5を下回る日の割合 {q*100:.1f}% → 17業種版の同等しきい値 {th:.3f}")
H = [5, 20, 60, 120]
p = px.reindex(rot.index)
def fwd(i, h): return (p.iloc[i + 1 + h] / p.iloc[i + 1] - 1) * 100 if i + 1 + h < len(p) else np.nan
base = {h: np.nanmean([fwd(i, h) for i in range(len(p))]) for h in H}
ev = np.where((rot < th) & (rot.shift() >= th))[0]
rows = [{"日付": rot.index[i].date(), **{f"{h}日": fwd(i, h) for h in H}} for i in ev]
df = pd.DataFrame(rows).set_index("日付")
print(f"\n−しきい値を下回った翌日に買う: {len(df)}回")
print(df.round(1).to_string())
print("\n平均 " + "  ".join(f"{h}日 {df[f'{h}日'].mean():+.1f}%(勝率{(df[f'{h}日'].dropna()>0).mean()*100:.0f}%)" for h in H))
print("無条件 " + "  ".join(f"{h}日 {base[h]:+.1f}%" for h in H))
rng = np.random.default_rng(0)
for h in [20, 60]:
    allf = pd.Series([fwd(i, h) for i in range(len(p))]).dropna()
    e = df[f"{h}日"].dropna()
    sims = np.array([allf.sample(len(e), random_state=int(rng.integers(1e9))).mean() for _ in range(5000)])
    print(f"{h}日: ランダムな日に同じ回数買って、この平均を上回る確率 p={(sims >= e.mean()).mean():.3f}")
