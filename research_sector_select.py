# -*- coding: utf-8 -*-
"""
============================================================
 エッジ探索②：セクター選択（どのセクターを持つか）の検証
============================================================
 research_edge.py（日経を持つ/降りるのタイミング）は全ルールがB&Hに負けた。
 ここでは「常に株を持ち、どのセクターを持つか」をダッシュボードの指標で選ぶ。

 ・毎週（または毎月）最終営業日の終値で判定 → 翌営業日終値で入れ替え
 ・上位K セクターのバスケット（構成銘柄の等金額）を等金額で保有
 ・比較対象: 全24セクター等金額（=ダッシュボードの「市場平均」）
 ・コスト: 入れ替えた金額に片道0.1%
 ・IS 〜2024-06 / OOS 2024-07〜
 ・ランダムにK セクターを選ぶ試行を1000回行い、偶然で出る超過の分布と比較
 注意: 構成銘柄は「現在の日経225」固定（生存者バイアスあり＝過去ほど強い銘柄が残っている）
============================================================
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dashboard import load_baskets, causal_gaussian, PRICES, FLOW_SIGMA_T  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

COST = 0.10
SPLIT = pd.Timestamp("2024-07-01")
START = pd.Timestamp("2022-06-01")  # 250日モメンタムのウォームアップ後


def load():
    baskets, labels, risk_off = load_baskets()
    px = pd.read_parquet(PRICES)
    close = px.pivot_table(index="Date", columns="Code", values="AdjC").sort_index().ffill(limit=3)
    sret = close.pct_change()
    bret = pd.DataFrame({k: sret[[c for c in v if c in sret.columns]].mean(axis=1)
                         for k, v in baskets.items()})              # 単純リターン
    lret = np.log1p(bret)
    rel = lret.sub(lret.mean(axis=1), axis=0) * 100
    flow = causal_gaussian(rel.fillna(0), FLOW_SIGMA_T)
    return bret, rel, flow, labels


def signals(rel, flow):
    s = {}
    for L in [20, 60, 120, 250]:
        s[f"モメンタム{L}日（上位）"] = rel.rolling(L).sum()
        s[f"リバーサル{L}日（下位）"] = -rel.rolling(L).sum()
    s["モメンタム250日・直近20日除く"] = rel.rolling(230).sum().shift(20)
    s["フロー強い順（ダッシュボード）"] = flow
    s["フロー弱い順（逆張り）"] = -flow
    s["フロー上向き転換（5日変化）"] = flow - flow.shift(5)
    s["120日強×フロー上向き"] = rel.rolling(120).sum().rank(axis=1) + (flow - flow.shift(5)).rank(axis=1)
    s["120日弱×フロー上向き（出遅れ反転）"] = (-rel.rolling(120).sum()).rank(axis=1) + (flow - flow.shift(5)).rank(axis=1)
    return s


def backtest(score, bret, K, freq):
    dates = bret.index
    reb = pd.Series(dates, index=dates).groupby(dates.to_period(freq)).last().values
    w = pd.DataFrame(np.nan, index=dates, columns=bret.columns)
    for d in reb:
        sc = score.loc[d].dropna()
        if len(sc) < K:
            continue
        top = sc.nlargest(K).index
        row = pd.Series(0.0, index=bret.columns)
        row[top] = 1.0 / K
        w.loc[d] = row
    w = w.ffill().fillna(0)
    held = w.shift(2).fillna(0)        # 判定日の翌日終値で執行 → 翌々日のリターンから反映
    turn = w.shift(1).fillna(0).diff().abs().sum(axis=1).fillna(0)
    r = (held * bret.fillna(0)).sum(axis=1) - turn * COST / 100
    return r, held


def stats(r, bench, mask):
    r, b = r[mask], bench[mask]
    ex = r - b
    yrs = len(r) / 245
    cagr = (1 + r).prod() ** (1 / yrs) - 1
    bcagr = (1 + b).prod() ** (1 / yrs) - 1
    te = ex.std() * np.sqrt(245)
    ir = (ex.mean() * 245) / te if te > 0 else np.nan
    return dict(年率=cagr * 100, 市場=bcagr * 100, 超過年率=(cagr - bcagr) * 100, IR=ir)


def main():
    bret, rel, flow, labels = load()
    bench = bret.mean(axis=1)
    dates = bret.index
    live = dates >= START
    is_m, oos_m = live & (dates < SPLIT), dates >= SPLIT
    rng = np.random.default_rng(0)

    rows = []
    for freq, fl in [("W", "週次"), ("M", "月次")]:
        for K in [3, 5]:
            # ランダム選択の超過分布（偶然の幅）
            rand = []
            for _ in range(300):
                sc = pd.DataFrame(rng.random(bret.shape), index=dates, columns=bret.columns)
                r, _ = backtest(sc, bret, K, freq)
                rand.append(stats(r, bench, live)["超過年率"])
            p95 = np.percentile(rand, 95)
            for name, sc in signals(rel, flow).items():
                r, held = backtest(sc, bret, K, freq)
                a, i, o = stats(r, bench, live), stats(r, bench, is_m), stats(r, bench, oos_m)
                pct = (np.array(rand) < a["超過年率"]).mean() * 100
                rows.append({"頻度": fl, "K": K, "ルール": name,
                             "全期間超過/年": a["超過年率"], "IS超過/年": i["超過年率"],
                             "OOS超過/年": o["超過年率"], "IR": a["IR"],
                             "ランダム比順位%": pct, "ランダム95%": p95})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.round(2).sort_values(["頻度", "K", "全期間超過/年"], ascending=[True, True, False])
          .to_string(index=False))
    df.to_csv(Path(__file__).parent / "out" / "sector_select_result.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
