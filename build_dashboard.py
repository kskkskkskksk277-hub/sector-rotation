# -*- coding: utf-8 -*-
"""
============================================================
 セクターローテーション：ダッシュボード生成
============================================================
 data/prices.parquet からバスケット別の相対強弱と資金フローを計算し、
 out/index.html（スマホ対応・Plotly）を出力します。

 指標の定義：
 ・バスケット日次リターン = 構成銘柄の対数リターンの単純平均
 ・相対リターン rel = バスケットリターン − 全バスケット平均（市場）
 ・累積相対強弱 RS = rel の累積和（市場に対してどれだけ勝ち越しているか）
 ・資金フロー flow = rel をガウス平滑化したもの（%/日）
     プラス＝そのバスケットへ資金流入、マイナス＝流出
 ・ローテーション指数 = 攻めバスケットの flow 平均 − 守りバスケットの flow 平均
     プラス＝リスクオン（景気敏感・グロースへ資金）、マイナス＝守りへ退避
============================================================
"""
from pathlib import Path
import datetime as dt
import json
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter, gaussian_filter1d

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
PRICES = ROOT / "data" / "prices.parquet"
INDICES = ROOT / "data" / "indices.parquet"
OUT_DIR = ROOT / "out"

# --- 計算パラメータ ---------------------------------------------------
FLOW_SIGMA_T = 4        # フローの時間方向の平滑化（営業日）
HEAT_SIGMA_B = 0.8      # ヒートマップのバスケット方向の平滑化
SIGMA_WINDOW = 120      # ±σバンドを測る期間（営業日）
WARMUP = 60             # 表示から捨てる先頭期間（平滑化が安定するまで）
ACCEL_TH = 1.2          # 加速/減速シグナルのしきい値（σ単位）

# dataviz検証済み（12色・CVDセーフ順）
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948",
           "#e87ba4", "#eb6834", "#0d9ddb", "#7a9c00", "#a24bcf", "#946200"]


def load_baskets():
    cfg = json.loads((ROOT / "baskets.json").read_text(encoding="utf-8"))
    baskets = {name: [c + "0" for c in b["codes"]] for name, b in cfg["baskets"].items()}
    labels = {name: b["label"] for name, b in cfg["baskets"].items()}
    return baskets, labels, cfg["risk_off"]


def compute():
    baskets, labels, risk_off = load_baskets()
    px = pd.read_parquet(PRICES)
    close = px.pivot_table(index="Date", columns="Code", values="AdjC").sort_index()
    close = close.ffill(limit=3)
    ret = np.log(close).diff() * 100.0  # %表記の対数リターン

    basket_ret = pd.DataFrame({
        name: ret[[c for c in codes if c in ret.columns]].mean(axis=1)
        for name, codes in baskets.items()
    })
    market = basket_ret.mean(axis=1)
    rel = basket_ret.sub(market, axis=0).dropna(how="all")

    rs = rel.cumsum()
    flow = pd.DataFrame(
        gaussian_filter1d(rel.to_numpy(), sigma=FLOW_SIGMA_T, axis=0, mode="nearest"),
        index=rel.index, columns=rel.columns)

    risk_on = [b for b in flow.columns if b not in risk_off]
    rot = flow[risk_on].mean(axis=1) - flow[risk_off].mean(axis=1)

    sigma = rot.rolling(SIGMA_WINDOW, min_periods=SIGMA_WINDOW // 2).std()

    # --- シグナル検出 ---
    sign = np.sign(rot)
    cross_up = (sign > 0) & (sign.shift() <= 0)
    cross_dn = (sign < 0) & (sign.shift() >= 0)
    plus_break = (rot > sigma) & (rot.shift() <= sigma.shift())
    minus_break = (rot < -sigma) & (rot.shift() >= -sigma.shift())

    accel = rot.diff().diff()  # 傾きの変化（2階差分）
    accel_s = pd.Series(gaussian_filter1d(accel.fillna(0).to_numpy(), 2), index=rot.index)
    ath = accel_s.rolling(SIGMA_WINDOW, min_periods=SIGMA_WINDOW // 2).std() * ACCEL_TH
    accel_alert = (accel_s > ath) & (accel_s.shift() <= ath.shift())
    decel_alert = (accel_s < -ath) & (accel_s.shift() >= -ath.shift())

    # --- 参考指数（TOPIX・日経平均）: 表示開始日=100として指数化 ---
    if INDICES.exists():
        mkt = pd.read_parquet(INDICES).set_index("Date").sort_index()
        mkt = mkt.reindex(rel.index).ffill()
    else:
        mkt = pd.DataFrame(index=rel.index, columns=["TOPIX", "NIKKEI225"], dtype=float)

    # ウォームアップ期間を落とす
    keep = rel.index[WARMUP:]
    frames = dict(rs=rs, flow=flow)
    series = dict(rot=rot, sigma=sigma, cross_up=cross_up, cross_dn=cross_dn,
                  plus_break=plus_break, minus_break=minus_break,
                  accel=accel_alert, decel=decel_alert)
    frames = {k: v.loc[keep] for k, v in frames.items()}
    series = {k: v.loc[keep] for k, v in series.items()}

    mkt = mkt.loc[keep]
    for col in mkt.columns:
        base = mkt[col].dropna()
        if not base.empty:
            mkt[col] = mkt[col] / base.iloc[0] * 100.0
    frames["mkt_idx"] = mkt

    return frames, series, labels


def sig_trace(rot, mask, name, symbol, color, size=8):
    idx = mask[mask].index
    return go.Scatter(x=idx, y=rot.loc[idx], mode="markers", name=name,
                      marker=dict(symbol=symbol, color=color, size=size,
                                  line=dict(width=1, color="white")),
                      hovertemplate=f"{name}<br>%{{x|%Y-%m-%d}}<extra></extra>")


def range_buttons():
    return dict(
        rangeselector=dict(
            buttons=[dict(count=3, label="3M", step="month", stepmode="backward"),
                     dict(count=6, label="6M", step="month", stepmode="backward"),
                     dict(count=1, label="1Y", step="year", stepmode="backward"),
                     dict(step="all", label="All")],
            x=0, y=1.15))


def build_figs(frames, series, labels):
    rot, sigma = series["rot"], series["sigma"]

    # --- 図1: ローテーション指数 ---
    f1 = go.Figure()
    f1.add_hline(y=0, line=dict(color="#d64550", width=1))
    f1.add_trace(go.Scatter(x=sigma.index, y=sigma, name="+σ", line=dict(color="#9aa0a6", dash="dot", width=1)))
    f1.add_trace(go.Scatter(x=sigma.index, y=-sigma, name="−σ", line=dict(color="#9aa0a6", dash="dot", width=1)))
    f1.add_trace(go.Scatter(x=rot.index, y=rot, name="ローテーション指数",
                            line=dict(color="#1f2430", width=2),
                            hovertemplate="%{x|%Y-%m-%d}<br>指数: %{y:.3f}<extra></extra>"))
    f1.add_trace(sig_trace(rot, series["cross_up"], "ゼロクロス↑（リスクオン転換）", "circle", "#2f9e44"))
    f1.add_trace(sig_trace(rot, series["cross_dn"], "ゼロクロス↓（リスクオフ転換）", "circle", "#d64550"))
    f1.add_trace(sig_trace(rot, series["plus_break"], "+σブレイク（過熱気味）", "circle", "#3b6fd4"))
    f1.add_trace(sig_trace(rot, series["minus_break"], "−σブレイク（悲観極まる）", "circle", "#8a3ffc"))
    f1.add_trace(sig_trace(rot, series["accel"], "加速アラート", "triangle-up", "#2f9e44", 9))
    f1.add_trace(sig_trace(rot, series["decel"], "減速アラート", "triangle-down", "#d64550", 9))

    # --- 同じグラフに日経平均・TOPIXを右軸で重ねる（表示開始日=100） ---
    mkt = frames["mkt_idx"]
    idx_names = {"NIKKEI225": "日経平均", "TOPIX": "TOPIX"}
    idx_colors = {"NIKKEI225": "#eb6834", "TOPIX": "#2a78d6"}
    idx_trace_pos = []
    for col in ["NIKKEI225", "TOPIX"]:
        if col not in mkt.columns or mkt[col].dropna().empty:
            continue
        idx_trace_pos.append(len(f1.data))
        f1.add_trace(go.Scatter(
            x=mkt.index, y=mkt[col], name=idx_names[col], yaxis="y2",
            line=dict(color=idx_colors[col], width=1.6), opacity=0.75,
            hovertemplate=idx_names[col] + "<br>%{x|%Y-%m-%d}<br>指数値: %{y:.1f}（開始日=100）<extra></extra>"))

    f1.update_layout(
        xaxis=range_buttons(), height=460,
        yaxis=dict(title=dict(text="ローテーション指数", font=dict(size=11))),
        yaxis2=dict(title=dict(text="日経平均・TOPIX（開始=100）", font=dict(size=11)),
                    overlaying="y", side="right", showgrid=False),
        updatemenus=[dict(
            type="buttons", direction="right",
            x=1, xanchor="right", y=1.18, yanchor="top",
            pad=dict(r=0, t=0), font=dict(size=11),
            buttons=[dict(
                label="日経・TOPIX 表示/非表示",
                method="restyle",
                args=[{"visible": True}, idx_trace_pos],
                args2=[{"visible": "legendonly"}, idx_trace_pos])])])

    # --- 図2: 資金フロー地形図 ---
    flow = frames["flow"]
    z = gaussian_filter(flow.to_numpy().T, sigma=(HEAT_SIGMA_B, 0), mode="nearest")
    zmax = np.percentile(np.abs(z), 98)
    f2 = go.Figure(go.Contour(
        z=z, x=flow.index, y=[labels[c] for c in flow.columns],
        colorscale="RdBu", reversescale=True, zmid=0, zmin=-zmax, zmax=zmax,
        line=dict(color="rgba(30,34,45,0.55)", width=1),
        contours=dict(coloring="heatmap"),
        colorbar=dict(title="流入(+)<br>流出(−)", thickness=12),
        hovertemplate="%{y}<br>%{x|%Y-%m-%d}<br>フロー: %{z:.3f} %/日<extra></extra>"))
    f2.update_layout(xaxis=range_buttons(), height=500)

    # --- 図3: 累積相対強弱 ---
    rs = frames["rs"]
    last = rs.iloc[-1].abs().sort_values(ascending=False)
    visible_top = set(last.index[:4])
    f3 = go.Figure()
    for i, col in enumerate(rs.columns):
        f3.add_trace(go.Scatter(
            x=rs.index, y=rs[col], name=labels[col],
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            visible=True if col in visible_top else "legendonly",
            hovertemplate=labels[col] + "<br>%{x|%Y-%m-%d}<br>市場比: %{y:.1f}%<extra></extra>"))
    f3.update_layout(xaxis=range_buttons(), height=450)

    for f in (f1, f2, f3):
        f.update_layout(
            template="plotly_white", font=dict(family="'Helvetica Neue',Arial,'Hiragino Sans','Yu Gothic',sans-serif", size=12),
            margin=dict(l=10, r=10, t=45, b=10),
            legend=dict(orientation="h", y=-0.12, font=dict(size=10)),
            hoverlabel=dict(font_size=12))
    return f1, f2, f3


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>日経225 セクターローテーション</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  body {{ margin: 0; background: #fafbfc; color: #1f2430;
         font-family: 'Helvetica Neue', Arial, 'Hiragino Sans', 'Yu Gothic', sans-serif; }}
  .wrap {{ max-width: 1100px; margin: 0 auto; padding: 12px; }}
  h1 {{ font-size: 1.25rem; margin: 8px 0 2px; }}
  .updated {{ color: #6b7280; font-size: .8rem; margin-bottom: 10px; }}
  .card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 10px;
           padding: 6px; margin-bottom: 14px; }}
  .card h2 {{ font-size: 1rem; margin: 8px 10px 0; }}
  .card p.note {{ color: #6b7280; font-size: .75rem; margin: 2px 10px 0; }}
  details {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 10px;
             padding: 10px 14px; margin-bottom: 14px; font-size: .85rem; line-height: 1.7; }}
  summary {{ cursor: pointer; font-weight: 600; }}
  .js-plotly-plot {{ width: 100%; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>日経225 セクターローテーション</h1>
  <div class="updated">最終更新: {updated}（毎営業日 夜に自動更新）</div>
  <details>
    <summary>この指標の読み方</summary>
    <ul>
      <li><b>ローテーション指数</b>: 攻めのバスケット（半導体AI・銀行・商社など）と守りのバスケット（ディフェンシブ・通信）の資金フローの差。プラス圏＝リスクオン、マイナス圏＝リスクオフ。</li>
      <li><b>ゼロクロス</b>: 攻守の入れ替わりのタイミング。<b>±σブレイク</b>: 偏りが普段の変動幅を超えた（行き過ぎ）サイン。</li>
      <li><b>日経平均・TOPIX（右軸）</b>: 実際の相場の値動き（表示開始日を100とした指数）。ローテーション指数がプラスなのに相場が下げている、といったズレを確認できます。右上のボタンでまとめて表示/非表示を切り替えられます。</li>
      <li><b>地形図</b>: 各バスケットへの資金の流入（赤）・流出（青）。縦に見ると「今どこが買われているか」、横に見ると「そのテーマがいつから続いているか」。</li>
      <li><b>累積相対強弱</b>: 市場平均に対する勝ち負けの積み上げ。右肩上がり＝市場より強い。</li>
      <li>数値はすべて株価から計算した加工済みの独自指標です。</li>
    </ul>
  </details>
  <div class="card"><h2>ローテーション指数（＋＝リスクオン ／ −＝リスクオフ）</h2>
    <p class="note">細い線は日経平均・TOPIX（右軸・開始=100）。右上のボタンでまとめて表示/非表示</p>{fig1}</div>
  <div class="card"><h2>資金フロー地形図（赤＝流入・青＝流出）</h2>{fig2}</div>
  <div class="card"><h2>累積相対強弱（市場平均に対する勝ち負け・%）</h2>
    <p class="note">凡例タップで表示/非表示を切り替え（初期表示は直近の動きが大きい4つ）</p>{fig3}</div>
  <details>
    <summary>各セクターの採用銘柄一覧</summary>
    {baskets_html}
  </details>
</div>
</body>
</html>
"""


def baskets_table_html() -> str:
    """採用銘柄一覧（銘柄名と証券コードのみ＝公開情報。株価データは含まない）"""
    cfg = json.loads((ROOT / "baskets.json").read_text(encoding="utf-8"))
    parts = []
    for name, b in cfg["baskets"].items():
        members = "、".join(f"{co}({code})" for code, co in b["codes"].items())
        group = "守り" if name in cfg["risk_off"] else "攻め"
        parts.append(f'<p style="margin:6px 0"><b>{b["label"]}</b>'
                     f'<span style="color:#6b7280">（{group}・{len(b["codes"])}銘柄）</span><br>{members}</p>')
    return "\n".join(parts)


def main() -> None:
    if not PRICES.exists():
        sys.exit("data/prices.parquet がありません。先に fetch_data.py を実行してください。")
    frames, series, labels = compute()
    f1, f2, f3 = build_figs(frames, series, labels)

    cfg = {"responsive": True, "displaylogo": False,
           "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}
    parts = [f.to_html(full_html=False, include_plotlyjs=False, config=cfg)
             for f in (f1, f2, f3)]

    jst = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=9)
    html = HTML_TEMPLATE.format(updated=jst.strftime("%Y-%m-%d %H:%M JST"),
                                fig1=parts[0], fig2=parts[1], fig3=parts[2],
                                baskets_html=baskets_table_html())
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"出力完了: {OUT_DIR / 'index.html'}（データ最終日: {series['rot'].index[-1]:%Y-%m-%d}）")


if __name__ == "__main__":
    main()
