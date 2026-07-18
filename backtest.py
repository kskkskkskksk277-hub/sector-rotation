# -*- coding: utf-8 -*-
"""
============================================================
 シグナルのバックテスト（イベントスタディ）
============================================================
 ダッシュボードの6種シグナル（ゼロクロス↑↓・±σブレイク・加速/減速）の
 発生後に各資産を買った場合の 1ヶ月/3ヶ月/6ヶ月 先のリターンを検証し、
 Excel（サマリー＋取引一覧）に出力します。

 検証対象の資産:
   ・日経平均 / TOPIX
   ・攻めバスケット平均（リスクオン17セクターの等ウェイト）
   ・守りバスケット平均（リスクオフ4セクター）
   ・フロー最弱3セクター（シグナル日に資金流出が最も強い3つ）＝逆張り
   ・フロー最強3セクター（流入が最も強い3つ）＝順張り

 重要な前提（結果の読み方に影響）:
   ・エントリーは「シグナル日の5営業日後の終値」。表示上のシグナルは
     前後数日を使う平滑化で確定するため、当日には確定していない。
     実運用で再現できるタイミングに寄せるための遅延。
   ・売買コスト・税・配当は考慮しない
   ・シグナル同士の期間が重なることがある（各件は独立ではない）
============================================================
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dashboard import compute, load_baskets, PRICES  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
INDICES = ROOT / "data" / "indices.parquet"
OUT_XLSX = Path(r"C:\Users\kawai\OneDrive\デスクトップ\株式投資\分析結果") / \
    "セクターローテーション_シグナルバックテスト.xlsx"

ENTRY_LAG = 5                      # シグナル日→エントリーまでの営業日数
HORIZONS = {"1ヶ月": 20, "3ヶ月": 60, "6ヶ月": 120}   # 営業日

SIGNAL_LABELS = {
    "cross_up": "ゼロクロス↑（リスクオン転換）",
    "cross_dn": "ゼロクロス↓（リスクオフ転換）",
    "plus_break": "+σブレイク（過熱気味）",
    "minus_break": "−σブレイク（悲観極まる）",
    "accel": "加速アラート",
    "decel": "減速アラート",
}


def basket_logret() -> pd.DataFrame:
    """バスケット別の日次対数リターン（%）— build_dashboard と同じ計算"""
    baskets, labels, risk_off = load_baskets()
    px = pd.read_parquet(PRICES)
    close = px.pivot_table(index="Date", columns="Code", values="AdjC").sort_index()
    close = close.ffill(limit=3)
    ret = np.log(close).diff() * 100.0
    basket_ret = pd.DataFrame({
        name: ret[[c for c in codes if c in ret.columns]].mean(axis=1)
        for name, codes in baskets.items()
    })
    return basket_ret, labels, risk_off


def build_asset_logrets() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """検証対象資産の日次対数リターン（%）と、セクター別リターンを返す"""
    basket_ret, labels, risk_off = basket_logret()
    risk_on = [c for c in basket_ret.columns if c not in risk_off]

    idx = pd.read_parquet(INDICES).set_index("Date").sort_index()
    idx_lr = np.log(idx).diff() * 100.0

    assets = pd.DataFrame({
        "日経平均": idx_lr["NIKKEI225"],
        "TOPIX": idx_lr["TOPIX"],
        "攻めバスケット平均": basket_ret[risk_on].mean(axis=1),
        "守りバスケット平均": basket_ret[risk_off].mean(axis=1),
    })
    assets = assets.reindex(basket_ret.index)
    return assets, basket_ret, labels


def fwd_return(logret: pd.Series, pos: int, h: int) -> float:
    """pos日目の終値で買い、h営業日後の終値で売った単純リターン（%）"""
    window = logret.iloc[pos + 1: pos + 1 + h]
    if len(window) < h or window.isna().any():
        return np.nan
    return (np.exp(window.sum() / 100.0) - 1.0) * 100.0


def main() -> None:
    frames, series, labels = compute()
    flow = frames["flow"]
    assets, basket_ret, _ = build_asset_logrets()
    dates = flow.index

    trades = []
    for sig_key, sig_label in SIGNAL_LABELS.items():
        sig_dates = series[sig_key][series[sig_key]].index
        for d in sig_dates:
            i = dates.get_loc(d)
            entry_i = i + ENTRY_LAG
            if entry_i >= len(dates):
                continue
            entry_d = dates[entry_i]

            # シグナル日のフローで最弱/最強3セクターを選ぶ
            f = flow.loc[d].sort_values()
            weakest3, strongest3 = list(f.index[:3]), list(f.index[-3:])

            targets = {
                "日経平均": assets["日経平均"],
                "TOPIX": assets["TOPIX"],
                "攻めバスケット平均": assets["攻めバスケット平均"],
                "守りバスケット平均": assets["守りバスケット平均"],
                "フロー最弱3セクター（逆張り）": basket_ret[weakest3].mean(axis=1),
                "フロー最強3セクター（順張り）": basket_ret[strongest3].mean(axis=1),
            }
            for asset_name, lr in targets.items():
                lr = lr.reindex(dates)
                pos = dates.get_loc(entry_d)
                row = {
                    "シグナル": sig_label,
                    "資産": asset_name,
                    "シグナル日": d.date(),
                    "エントリー日": entry_d.date(),
                }
                if "最弱" in asset_name:
                    row["選ばれたセクター"] = "、".join(labels[c] for c in weakest3)
                elif "最強" in asset_name:
                    row["選ばれたセクター"] = "、".join(labels[c] for c in strongest3)
                else:
                    row["選ばれたセクター"] = ""
                for h_label, h in HORIZONS.items():
                    row[h_label] = fwd_return(lr, pos, h)
                trades.append(row)

    df = pd.DataFrame(trades)

    # ベースライン: 全営業日で同じ保有をした場合の平均（シグナルに意味があるかの比較用）
    base = {}
    for asset_name in ["日経平均", "TOPIX", "攻めバスケット平均", "守りバスケット平均"]:
        lr = assets[asset_name].reindex(dates)
        for h_label, h in HORIZONS.items():
            vals = [fwd_return(lr, i, h) for i in range(len(dates))]
            vals = [v for v in vals if not np.isnan(v)]
            base[(asset_name, h_label)] = (np.mean(vals),
                                           np.mean([v > 0 for v in vals]) * 100)

    write_excel(df, base)

    # コンソールにも要約を出す
    print(f"取引数: {len(df)}（シグナル{df['シグナル'].nunique()}種 × 資産6 × 発生回数）")
    for h_label in HORIZONS:
        pivot = df.pivot_table(values=h_label, index="シグナル", columns="資産", aggfunc="mean")
        print(f"\n===== {h_label}後の平均リターン（%）=====")
        print(pivot.round(2).to_string())


def write_excel(df: pd.DataFrame, base: dict) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    bold = Font(name="Arial", bold=True)
    normal = Font(name="Arial")
    hdr_fill = PatternFill("solid", fgColor="DDEBF7")
    note_font = Font(name="Arial", size=9, color="666666")

    # ---------- シート2: 取引一覧（先に作って行数を確定させる） ----------
    ws2 = wb.create_sheet("取引一覧")
    cols = ["シグナル", "資産", "シグナル日", "エントリー日", "選ばれたセクター",
            "1ヶ月", "3ヶ月", "6ヶ月"]
    for j, c in enumerate(cols, 1):
        cell = ws2.cell(1, j, c)
        cell.font = bold
        cell.fill = hdr_fill
    for i, (_, row) in enumerate(df.iterrows(), 2):
        for j, c in enumerate(cols, 1):
            v = row[c]
            if c in ("1ヶ月", "3ヶ月", "6ヶ月"):
                if pd.isna(v):
                    continue
                cell = ws2.cell(i, j, float(v) / 100.0)   # %は小数で保持
                cell.number_format = "0.0%"
            else:
                cell = ws2.cell(i, j, str(v))
            cell.font = normal
    n = len(df) + 1
    widths2 = [30, 26, 12, 12, 40, 9, 9, 9]
    for j, w in enumerate(widths2, 1):
        ws2.column_dimensions[get_column_letter(j)].width = w

    # ---------- シート1: サマリー ----------
    ws = wb.active
    ws.title = "サマリー"
    ws["A1"] = "セクターローテーション・シグナル バックテスト"
    ws["A1"].font = Font(name="Arial", bold=True, size=13)
    period = f"{df['シグナル日'].min()} 〜 {df['シグナル日'].max()}"
    notes = [
        f"検証期間: {period}（シグナル発生日ベース・約5年）",
        f"エントリー: シグナル日の5営業日後の終値（シグナルは平滑化の関係で当日には確定しないため）",
        "リターン: エントリー終値から20/60/120営業日後（≒1/3/6ヶ月）の終値まで。コスト・税・配当は未考慮",
        "勝率: リターンがプラスだった割合。件数が少ないシグナルは偶然の影響が大きい点に注意",
        "シグナル同士の保有期間が重なることがあるため、各件は完全に独立した取引ではない",
        "サマリーの数値は「取引一覧」シートから数式で集計（取引一覧を編集すると自動で再計算される）",
    ]
    for k, t in enumerate(notes, 2):
        ws.cell(k, 1, "・" + t).font = note_font

    r0 = len(notes) + 3
    headers = ["シグナル", "資産", "件数",
               "1ヶ月平均", "1ヶ月勝率", "3ヶ月平均", "3ヶ月勝率", "6ヶ月平均", "6ヶ月勝率"]
    for j, h in enumerate(headers, 1):
        cell = ws.cell(r0, j, h)
        cell.font = bold
        cell.fill = hdr_fill
        cell.alignment = Alignment(horizontal="center")

    sig_order = list(SIGNAL_LABELS.values())
    asset_order = ["日経平均", "TOPIX", "攻めバスケット平均", "守りバスケット平均",
                   "フロー最弱3セクター（逆張り）", "フロー最強3セクター（順張り）"]
    hcols = {"1ヶ月": "F", "3ヶ月": "G", "6ヶ月": "H"}

    r = r0 + 1
    for sig in sig_order:
        for asset in asset_order:
            ws.cell(r, 1, sig).font = normal
            ws.cell(r, 2, asset).font = normal
            crit = f'取引一覧!$A$2:$A${n},$A{r},取引一覧!$B$2:$B${n},$B{r}'
            ws.cell(r, 3, f'=COUNTIFS({crit})').font = normal
            for j, (h_label, col) in enumerate(hcols.items()):
                rng = f'取引一覧!${col}$2:${col}${n}'
                avg = ws.cell(r, 4 + j * 2,
                              f'=IFERROR(AVERAGEIFS({rng},{crit}),"")')
                avg.number_format = "+0.0%;-0.0%;0.0%"
                avg.font = normal
                win = ws.cell(r, 5 + j * 2,
                              f'=IFERROR(COUNTIFS({crit},{rng},">0")/'
                              f'COUNTIFS({crit},{rng},"<>"),"")')
                win.number_format = "0%"
                win.font = normal
            r += 1
        r += 1  # シグナル間に空行

    # ベースライン表
    r += 1
    ws.cell(r, 1, "【ベースライン】シグナルに関係なく毎日買った場合の平均（比較用・Pythonで全営業日から算出）").font = bold
    r += 1
    for j, h in enumerate(["資産", "", "",
                           "1ヶ月平均", "1ヶ月勝率", "3ヶ月平均", "3ヶ月勝率", "6ヶ月平均", "6ヶ月勝率"], 1):
        if h:
            cell = ws.cell(r, j, h)
            cell.font = bold
            cell.fill = hdr_fill
    r += 1
    for asset_name in ["日経平均", "TOPIX", "攻めバスケット平均", "守りバスケット平均"]:
        ws.cell(r, 1, asset_name).font = normal
        for j, h_label in enumerate(HORIZONS):
            mean_v, win_v = base[(asset_name, h_label)]
            c1 = ws.cell(r, 4 + j * 2, mean_v / 100.0)
            c1.number_format = "+0.0%;-0.0%;0.0%"
            c1.font = normal
            c2 = ws.cell(r, 5 + j * 2, win_v / 100.0)
            c2.number_format = "0%"
            c2.font = normal
        r += 1

    widths = [30, 26, 7, 11, 10, 11, 10, 11, 10]
    for j, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.freeze_panes = ws.cell(r0 + 1, 1)

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT_XLSX)
    print(f"Excel出力: {OUT_XLSX}")


if __name__ == "__main__":
    main()
