import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/kawai/sector-rotation")
from build_dashboard import causal_gaussian
sys.stdout.reconfigure(encoding="utf-8")
d = pd.read_parquet(r"C:/Users/kawai/sector-rotation/research/t17.parquet")
d = d[d.index >= "2009-01-05"].ffill(limit=3)
r = d.pct_change()
r = r.mask((r > 0.3) | (r < -0.4), 0.0)  # 株式分割の未調整ジャンプを除去
# 異常値チェック
print("最大日次|r|:", r.abs().max().round(3).to_dict())
etf = r.drop(columns="1306.T"); topix = r["1306.T"]
bench = etf.mean(axis=1)
lr = np.log1p(etf); rel = lr.sub(lr.mean(axis=1), axis=0) * 100
flow = causal_gaussian(rel.fillna(0), 8)
COST = 0.2
def bt(score, K, freq="M"):
    dates = etf.index
    reb = pd.Series(dates, index=dates).groupby(dates.to_period(freq)).last().values
    w = pd.DataFrame(np.nan, index=dates, columns=etf.columns)
    for dd in reb:
        sc = score.loc[dd].dropna()
        if len(sc) < 15: continue
        row = pd.Series(0.0, index=etf.columns); row[sc.nlargest(K).index] = 1 / K; w.loc[dd] = row
    w = w.ffill().fillna(0)
    held = w.shift(2).fillna(0)
    turn = w.shift(1).fillna(0).diff().abs().sum(axis=1).fillna(0)
    return (held * etf.fillna(0)).sum(axis=1) - turn * COST / 100
def cagr(x): return ((1 + x).prod() ** (245 / len(x)) - 1) * 100
periods = [("2010-01", "2012-12"), ("2013-01", "2016-12"), ("2017-01", "2020-12"), ("2021-01", "2023-12"), ("2024-01", "2026-12")]
start = "2010-01-01"
sigs = {}
for L in [20, 60, 120, 250]:
    sigs[f"モメ{L}"] = rel.rolling(L).sum()
    sigs[f"リバ{L}"] = -rel.rolling(L).sum()
sigs["モメ250除20"] = rel.rolling(230).sum().shift(20)
sigs["フロー強"] = flow
sigs["フロー弱"] = -flow
sigs["フロー変化5d"] = flow - flow.shift(5)
rows = []
for K in [3, 5]:
    for n, s in sigs.items():
        x = bt(s, K)[start:]; b = bench[start:]
        ex = x - b
        row = {"K": K, "ルール": n, "年率": cagr(x), "等金額17": cagr(b), "超過": cagr(x) - cagr(b),
               "t値": ex.mean() / ex.std() * np.sqrt(len(ex)),
               "勝ち月%": (ex.resample("ME").sum() > 0).mean() * 100}
        for a, z in periods:
            row[a[:4] + "-" + z[2:4]] = cagr(x[a:z]) - cagr(b[a:z])
        rows.append(row)
pd.set_option("display.width", 250)
df = pd.DataFrame(rows).round(1)
print(df.to_string(index=False))
print("TOPIX(1306) 年率", round(cagr(topix[start:]), 1), " 等金額17", round(cagr(bench[start:]), 1))
