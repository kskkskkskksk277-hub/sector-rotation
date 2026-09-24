# -*- coding: utf-8 -*-
"""ローテーション指数がしきい値を跨いだ翌朝（日経平均の始値）に買った場合の、その後のリターン"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/kawai/sector-rotation")
sys.stdout.reconfigure(encoding="utf-8")
from build_dashboard import compute, INDICES
TH = float(sys.argv[1]) if len(sys.argv) > 1 else -0.5
frames, series, labels = compute()
rot = series["rot"]
nk = pd.read_parquet(INDICES).set_index("Date").sort_index()[["NK_O", "NIKKEI225"]].dropna()
nk = nk.reindex(rot.index).ffill()
o, c = nk["NK_O"], nk["NIKKEI225"]
H = [5, 20, 60, 120]
def fwd(i, h):  # i日目の翌朝始値で買い、h営業日後の終値まで
    if i + h >= len(c): return np.nan
    return (c.iloc[i + h] / o.iloc[i + 1] - 1) * 100
base = {h: np.nanmean([fwd(i, h) for i in range(len(c) - 1)]) for h in H}
down = (rot < TH) & (rot.shift() >= TH)
up = (rot >= TH) & (rot.shift() < TH)
pos = {d: i for i, d in enumerate(rot.index)}
for name, ev in [(f"① {TH}を下回った翌朝に買う", down), (f"② {TH}の下から上に戻った翌朝に買う", up)]:
    ds = ev[ev].index
    print(f"\n{name}（{len(ds)}回）")
    rows = []
    for d in ds:
        i = pos[d]
        rows.append({"シグナル日": d.date(), "指数": round(rot.iloc[i], 2), "買値(翌朝始値)": round(o.iloc[i + 1]) if i + 1 < len(o) else None,
                     **{f"{h}日後%": round(fwd(i, h), 1) for h in H}})
    df = pd.DataFrame(rows); print(df.to_string(index=False))
    print("  平均   " + "  ".join(f"{h}日後 {df[f'{h}日後%'].mean():+.1f}%" for h in H))
    print("  勝率   " + "  ".join(f"{h}日後 {(df[f'{h}日後%'].dropna() > 0).mean()*100:.0f}%" for h in H))
print("\n参考: いつ買っても（全営業日平均） " + "  ".join(f"{h}日後 {base[h]:+.1f}%" for h in H))
