# -*- coding: utf-8 -*-
"""
============================================================
 エッジ探索：日経ETFの超過リターンを狙う売買ルールの検証
============================================================
 ダッシュボードの指標（ローテーション指数・σバンド・各シグナル・
 セクター別フロー・ブレッド＝プラスフローのセクター比率）を組み合わせた
 ロング/キャッシュ切替ルールを、次の条件で検証する。

 ・シグナルは因果的（その日までのデータのみ）→ 当日終値で判定し翌日終値で執行
 ・売買コスト: 片道0.05%（ETFのスプレッド+手数料相当）
 ・学習期間（IS）: 2021-11〜2024-06 でルールを比較・選抜
 ・検証期間（OOS）: 2024-07〜 でその後を確認（ここで勝てないルールは過学習）
 ・比較対象: 日経平均の持ちっぱなし（Buy & Hold）
============================================================
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dashboard import compute, load_baskets, PRICES, INDICES  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

COST = 0.05          # 片道コスト（%）
SPLIT = "2024-07-01"  # IS/OOS の分割日


def load_all():
    frames, series, labels = compute()
    dates = frames["flow"].index
    idx = pd.read_parquet(INDICES).set_index("Date").sort_index()
    nk_ret = (np.log(idx["NIKKEI225"]).diff() * 100.0).reindex(dates).fillna(0)

    baskets, _, risk_off = load_baskets()
    flow = frames["flow"]
    rot, sigma = series["rot"], series["sigma"]

    ind = pd.DataFrame(index=dates)
    ind["rot"] = rot
    ind["sigma"] = sigma
    ind["def_flow"] = flow[[c for c in risk_off]].mean(axis=1)
    ind["semi_flow"] = flow["semiconductor_ai"]
    ind["breadth"] = (flow > 0).mean(axis=1)          # プラスフローのセクター比率
    for k in ["cross_up", "cross_dn", "plus_break", "minus_break", "accel", "decel"]:
        ind[k] = series[k].astype(bool)
    return ind, nk_ret, dates


def event_hold(ind, buy_key, hold):
    """イベント発生から hold 営業日ロング"""
    pos = pd.Series(0.0, index=ind.index)
    arr = ind[buy_key].to_numpy()
    until = -1
    for i in range(len(arr)):
        if arr[i]:
            until = i + hold
        if i <= until:
            pos.iloc[i] = 1.0
    return pos


def state_machine(ind, buy_key, sell_key):
    """buyイベントでロング、sellイベントでキャッシュ"""
    pos = pd.Series(0.0, index=ind.index)
    b, s = ind[buy_key].to_numpy(), ind[sell_key].to_numpy()
    p = 0.0
    for i in range(len(b)):
        if b[i]:
            p = 1.0
        elif s[i]:
            p = 0.0
        pos.iloc[i] = p
    return pos


def build_rules(ind):
    r = {}
    rot, sig = ind["rot"], ind["sigma"]
    r["R01 指数>0でロング"] = (rot > 0).astype(float)
    r["R02 指数>-σでロング（深いリスクオフのみ回避）"] = (rot > -sig).astype(float)
    r["R03 指数が5日前より上"] = (rot > rot.shift(5)).astype(float)
    r["R04 指数が10日前より上"] = (rot > rot.shift(10)).astype(float)
    r["R05 指数>0 または <-σ（押し目買い併用）"] = ((rot > 0) | (rot < -sig)).astype(float)
    r["R06 守りフロー<0（守りから資金流出中）"] = (ind["def_flow"] < 0).astype(float)
    r["R07 半導体フロー>0"] = (ind["semi_flow"] > 0).astype(float)
    r["R08 ブレッド>=50%"] = (ind["breadth"] >= 0.5).astype(float)
    r["R09 ブレッド>=40%"] = (ind["breadth"] >= 0.4).astype(float)
    r["R10 −σブレイク後60日ロング"] = event_hold(ind, "minus_break", 60)
    r["R11 −σブレイク後120日ロング"] = event_hold(ind, "minus_break", 120)
    r["R12 加速で買い減速で売り"] = state_machine(ind, "accel", "decel")
    r["R13 ゼロクロス↑買い↓売り"] = state_machine(ind, "cross_up", "cross_dn")
    r["R14 基本ロング、指数<0かつ5日下落で退避"] = (~((rot < 0) & (rot < rot.shift(5)))).astype(float)
    r["R15 基本ロング、指数<-σで退避（R02同値でない確認用）"] = (rot >= -sig).astype(float)
    r["R16 R14+押し目（<-σなら買い直す）"] = (
        (~((rot < 0) & (rot < rot.shift(5))) | (rot < -sig))).astype(float)
    r["R17 ブレッド>=35% or 指数<-σ"] = ((ind["breadth"] >= 0.35) | (rot < -sig)).astype(float)
    return r


def evaluate(pos, nk_ret, cost=COST):
    """pos: 当日終値で決めた目標ポジション → 翌日終値から反映"""
    held = pos.shift(2).fillna(0)     # 判定翌日の終値で執行＝リターン反映はさらに翌日から
    strat = nk_ret * held
    turn = pos.shift(1).fillna(0).diff().abs().fillna(0)
    strat = strat - turn * cost
    eq = np.exp(strat.cumsum() / 100.0)
    bh = np.exp(nk_ret.cumsum() / 100.0)
    dd = (eq / eq.cummax() - 1).min() * 100
    bh_dd = (bh / bh.cummax() - 1).min() * 100
    return {
        "リターン": (eq.iloc[-1] - 1) * 100,
        "B&H": (bh.iloc[-1] - 1) * 100,
        "超過": (eq.iloc[-1] - bh.iloc[-1]) * 100,
        "最大DD": dd, "B&H_DD": bh_dd,
        "売買回数": int((turn > 0).sum()),
        "滞在率": held.mean() * 100,
        "eq": eq, "strat": strat,
    }


def main() -> None:
    ind, nk_ret, dates = load_all()
    rules = build_rules(ind)
    split = pd.Timestamp(SPLIT)

    rows = []
    for name, pos in rules.items():
        full = evaluate(pos, nk_ret)
        is_ = evaluate(pos[dates < split], nk_ret[dates < split])
        oos = evaluate(pos[dates >= split], nk_ret[dates >= split])
        rows.append({
            "ルール": name,
            "IS超過": is_["超過"], "IS戦略": is_["リターン"], "IS_BH": is_["B&H"],
            "OOS超過": oos["超過"], "OOS戦略": oos["リターン"], "OOS_BH": oos["B&H"],
            "全期間超過": full["超過"], "最大DD": full["最大DD"],
            "売買回数": full["売買回数"], "滞在率": full["滞在率"],
        })
    df = pd.DataFrame(rows).sort_values("IS超過", ascending=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    print("===== 学習期間（〜2024/06）の超過リターン順 =====")
    print(df.round(1).to_string(index=False))

    # 年別リターン（上位ルールとB&H）
    print("\n===== 年別リターン（%） =====")
    top_names = df.head(5)["ルール"].tolist()
    yearly = {}
    bh_eq = np.exp(nk_ret.cumsum() / 100.0)
    for name in top_names + ["B&H"]:
        if name == "B&H":
            strat = nk_ret
        else:
            strat = evaluate(rules[name], nk_ret)["strat"]
        y = strat.groupby(strat.index.year).sum()
        yearly[name] = ((np.exp(y / 100) - 1) * 100).round(1)
    print(pd.DataFrame(yearly).to_string())


if __name__ == "__main__":
    main()
