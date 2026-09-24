# -*- coding: utf-8 -*-
"""
積立（毎月定額）vs 積立＋−σブレイク時の追加投入 の比較（TOPIX ETF 1306、2009-07〜）
・総投入額は全プラン同じ（毎月3万円）。追加投入プランは一部を現金で貯め、シグナル時に一括投入
・シグナルはTOPIX-17業種ETFから作ったローテーション指数（ダッシュボードと同じ計算）
・約定はシグナル翌営業日の終値。現金に利息はつけない（不利側に見積もる）
・「偶然でも同じ結果になるか」を見るため、同じ回数だけランダムな日に投入する試行と比べる
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/kawai/sector-rotation")
from build_dashboard import causal_gaussian, cooldown
sys.stdout.reconfigure(encoding="utf-8")

d = pd.read_parquet(r"C:/Users/kawai/sector-rotation/research/t17.parquet")
d = d[d.index >= "2009-01-05"].ffill(limit=3)
r = d.pct_change(); r = r.mask((r > 0.3) | (r < -0.4), 0.0)
price = (1 + r["1306.T"].fillna(0)).cumprod()          # 分割補正済みの基準価格
etf = r.drop(columns="1306.T")
lr = np.log1p(etf); rel = lr.sub(lr.mean(axis=1), axis=0) * 100
flow = causal_gaussian(rel.fillna(0), 8)
off = ["1617.T", "1621.T", "1627.T", "1628.T"]
rot = flow[[c for c in flow.columns if c not in off]].mean(axis=1) - flow[off].mean(axis=1)
sig = rot.rolling(120, min_periods=60).std()
mb = cooldown((rot < -sig) & (rot.shift() >= -sig.shift()))
pb = cooldown((rot > sig) & (rot.shift() <= sig.shift()))
MB = set(mb.shift(1, fill_value=False)[lambda s: s].index)   # 翌営業日に約定
PB = set(pb.shift(1, fill_value=False)[lambda s: s].index)

MONTHLY = 30000

def run(dates, reserve_ratio=0.0, deploy_days=frozenset(), skip_after_plus=False):
    """reserve_ratio: 毎月の積立のうち現金で貯めて追加投入に回す割合"""
    first = set(pd.Series(dates, index=dates).groupby(dates.to_period("M")).first())
    units, cash, paid, skip_until = 0.0, 0.0, 0.0, None
    for t in dates:
        p = price[t]
        if t in PB and skip_after_plus:
            skip_until = t + pd.Timedelta(days=30)    # 過熱後1か月は通常積立も現金へ
        if t in first:
            paid += MONTHLY
            if skip_until is not None and t <= skip_until:
                cash += MONTHLY
            else:
                units += MONTHLY * (1 - reserve_ratio) / p
                cash += MONTHLY * reserve_ratio
        if t in deploy_days and cash > 0:
            units += cash / p; cash = 0.0
    return units * price[dates[-1]] + cash, paid

def compare(start, end, label):
    dates = price[start:end].index
    base, paid = run(dates)
    print(f"\n■ {label}（{dates[0]:%Y/%m}〜{dates[-1]:%Y/%m}・投入総額 {paid/1e4:,.0f}万円）")
    print(f"  A 普通の積立（毎月3万円）          最終 {base/1e4:8,.1f}万円")
    mbd = [t for t in dates if t in MB]
    rng = np.random.default_rng(0)
    for ratio in [0.2, 0.5]:
        v, _ = run(dates, ratio, frozenset(mbd))
        # 同じ回数だけランダムな日に投入した場合（シグナルに意味がなければ同程度になる）
        sims = [run(dates, ratio, frozenset(rng.choice(dates, len(mbd), replace=False)))[0] for _ in range(300)]
        pct = (np.array(sims) < v).mean() * 100
        print(f"  B {int(MONTHLY*(1-ratio)/1e3)}千円積立＋{int(MONTHLY*ratio/1e3)}千円を貯めて−σ時投入  最終 {v/1e4:8,.1f}万円"
              f"（Aとの差 {(v-base)/1e4:+.1f}万円 / {(v/base-1)*100:+.2f}%）  ランダム投入に勝つ割合 {pct:.0f}%")
    v, _ = run(dates, 0.2, frozenset(mbd), skip_after_plus=True)
    print(f"  C Bの20%版＋過熱(+σ)後1か月は積立見送り   最終 {v/1e4:8,.1f}万円（Aとの差 {(v-base)/1e4:+.1f}万円 / {(v/base-1)*100:+.2f}%）")

compare("2009-07-01", None, "全期間 約17年")
for s, e in [("2009-07", "2014-06"), ("2012-01", "2016-12"), ("2015-01", "2019-12"),
             ("2018-01", "2022-12"), ("2021-10", "2026-09")]:
    compare(s, e, "5年間")
