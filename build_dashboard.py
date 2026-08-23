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
 ・資金フロー flow = rel を指数移動平均で平滑化したもの（%/日）
     プラス＝そのバスケットへ資金流入、マイナス＝流出
 ・ローテーション指数 = 攻めバスケットの flow 平均 − 守りバスケットの flow 平均
     プラス＝リスクオン（景気敏感・グロースへ資金）、マイナス＝守りへ退避

 【重要】平滑化はすべて「その日までのデータのみ」を使う後ろ向き（因果的）計算。
 中心化ガウス平滑（前後±8営業日を参照）を使っていた頃は、新しい日のデータが
 入るたびに過去のシグナル位置が動いてしまい、実運用では再現できない
 「未来を見た」指標になっていた。一度表示されたシグナルが動かないことを
 優先し、2026-07-18 に指数移動平均へ変更した（代償として反応は数日遅れる）。
============================================================
"""
from pathlib import Path
import datetime as dt
import json
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from scipy.signal import lfilter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
PRICES = ROOT / "data" / "prices.parquet"
INDICES = ROOT / "data" / "indices.parquet"
OUT_DIR = ROOT / "out"

# --- 計算パラメータ ---------------------------------------------------
# 片側（過去向き）ガウス平滑の σ。旧・中心化ガウス平滑（σ=4 / σ=2）と同じ
# ノイズ低減量になる値（片側なので σ は2倍）。未来のデータは一切参照しない。
FLOW_SIGMA_T = 8        # フローの時間方向の平滑化
ACCEL_SIGMA = 8         # 加速度（2階差分）の平滑化（2階差分はノイズが大きいため強めに）
COOLDOWN = 10           # 同一シグナルの再点灯を抑える日数（同じ局面の重複検出を1つにまとめる）
HEAT_SIGMA_B = 0.8      # ヒートマップのバスケット方向の平滑化（時間方向には掛けない）
SIGMA_WINDOW = 120      # ±σバンドを測る期間（営業日）
WARMUP = 60             # 表示から捨てる先頭期間（平滑化が安定するまで）
ACCEL_TH = 1.2          # 加速/減速シグナルのしきい値（σ単位）

# 21バスケット分の実線用カラー（隣接が似ないよう手動配列）
PALETTE = ["#2a78d6", "#e34948", "#008300", "#eda100", "#4a3aa7", "#1baf7a",
           "#eb6834", "#0d9ddb", "#a24bcf", "#7a9c00", "#e87ba4", "#946200",
           "#17877d", "#d43f8c", "#5a63e0", "#a05a2c", "#647087", "#4d7c0f",
           "#b0399e", "#c96f00", "#356ac0"]

# 全グラフ共通の余白（固定値にすることで横軸の位置を完全に揃える）
MARGIN = dict(l=84, r=70, t=45, b=10)

# 断面図・累積相対強弱で最初からチェックを入れておくセクター
DEFAULT_SECTORS = {
    "電力・ガス・鉄道", "不動産", "銀行", "保険", "建設・住宅", "海運",
    "資源・エネルギー", "商社", "半導体・AI", "ゲーム・エンタメ", "ネット・ITサービス",
}

# 地形図の縦軸用の短縮ラベル（左余白84pxに収める）
SHORT_LABEL = {
    "電力・ガス・鉄道": "電力ガス鉄道",
    "食品・生活用品": "食品・生活",
    "保険・証券・その他金融": "保険・金融",
    "建設・不動産・住宅": "建設・不動産",
    "資源・エネルギー": "エネルギー",
    "ネット・ITサービス": "ネット・IT",
    "ゲーム・エンタメ": "エンタメ",
    "機械・FA・ロボット": "機械・FA",
    "精密・医療機器": "精密・医療",
    "電機・電子部品": "電機・部品",
    "鉄鋼・非鉄金属": "鉄鋼・非鉄",
    "証券・その他金融": "証券・金融",
    "光デバイス・高速通信": "光デバイス",
    "AIソフト・エージェント": "AIソフト",
    "サイバーセキュリティ": "サイバーセキ",
    "量子コンピューター": "量子コンピュータ",
    "フィジカルAI・ロボット": "フィジカルAI",
    "サーバー冷却・空調": "冷却・空調",
    "レアアース・都市油田": "レアアース",
    "再生可能エネルギー": "再エネ",
    "ペロブスカイト太陽電池": "ペロブスカイト",
    "金利上昇メリット": "金利上昇メリ",
    "内需・円高メリット": "内需・円高",
    "SDV・車載ソフト": "SDV・車載",
    "バイオテクノロジー": "バイオ",
    "ディフェンシブ全体": "ディフェンシブ",
    "半導体部材・部品": "半導体部材",
    "半導体製造装置": "半導体製造装置",
    "SaaS・クラウド": "SaaS",
}


def short(label: str) -> str:
    return SHORT_LABEL.get(label, label)


def causal_gaussian(df, sigma: float, truncate: float = 3.0):
    """片側（過去向き）ガウス平滑。その日と過去 3σ 日だけを重み付き平均する。

    中心化ガウス平滑と違い未来を参照しないので、後からデータが増えても
    過去の値は一切変わらない。代わりに σ√(2/π) 日ぶん反応が遅れる。
    """
    n = int(truncate * sigma)
    k = np.exp(-0.5 * (np.arange(n + 1) / sigma) ** 2)
    k /= k.sum()                      # k[0]=当日, k[1]=前日, …
    arr = np.asarray(df, dtype=float)
    one_d = arr.ndim == 1
    if one_d:
        arr = arr[:, None]
    pad = np.repeat(arr[:1], n, axis=0)          # 先頭は最初の値で埋める
    out = lfilter(k, 1.0, np.vstack([pad, arr]), axis=0)[n:]
    if one_d:
        out = out[:, 0]
        return pd.Series(out, index=df.index)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def cooldown(mask: pd.Series, days: int = COOLDOWN) -> pd.Series:
    """点灯後 days 営業日は再点灯させない（同じ局面での連続検出を1つにまとめる）"""
    out = mask.to_numpy().copy()
    last = -10**9
    for i in np.flatnonzero(out):
        if i - last < days:
            out[i] = False
        else:
            last = i
    return pd.Series(out, index=mask.index)


def load_themes():
    """テーマ定義。codes を持つものと、基本層を合成する from_baskets 型がある"""
    path = ROOT / "themes.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))["themes"]


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

    # --- テーマ層（表示専用・銘柄が重複してよい。指数計算には一切使わない） ---
    # 市場平均は基本層のものを流用する。テーマ同士は重複するので独自の平均は取れない。
    themes = load_themes()
    theme_meta, theme_ret = {}, {}
    for key, t in themes.items():
        if "from_baskets" in t:
            codes = [c for b in t["from_baskets"] for c in baskets[b]]
        else:
            codes = [c + "0" if len(c) == 4 else c for c in t["codes"]]
        cols = [c for c in codes if c in ret.columns]
        if not cols:
            continue
        theme_meta[key] = {"label": t["label"], "group": t.get("group", "その他"),
                           "n": len(cols)}
        theme_ret[key] = ret[cols].mean(axis=1)
    theme_rel = pd.DataFrame(theme_ret).sub(market, axis=0).reindex(rel.index)
    theme_rs = theme_rel.cumsum()
    theme_flow = causal_gaussian(theme_rel.fillna(0), FLOW_SIGMA_T)

    rs = rel.cumsum()
    # 片側ガウス平滑（その日までのデータのみ）。過去の値が後から変わらない
    flow = causal_gaussian(rel, FLOW_SIGMA_T)

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
    accel_s = causal_gaussian(accel.fillna(0), ACCEL_SIGMA)
    ath = accel_s.rolling(SIGMA_WINDOW, min_periods=SIGMA_WINDOW // 2).std() * ACCEL_TH
    accel_alert = (accel_s > ath) & (accel_s.shift() <= ath.shift())
    decel_alert = (accel_s < -ath) & (accel_s.shift() >= -ath.shift())

    # --- 参考指数（TOPIX・日経平均）: 表示開始日=100として指数化 ---
    ohlc_cols = ["NK_O", "NK_H", "NK_L", "NIKKEI225"]
    if INDICES.exists():
        idx_raw = pd.read_parquet(INDICES).set_index("Date").sort_index()
        mkt = idx_raw[["TOPIX", "NIKKEI225"]].reindex(rel.index).ffill()
        # ローソク足用の四本値（ffillで偽の足を作らないよう欠損日はそのまま落とす）
        if set(ohlc_cols) <= set(idx_raw.columns):
            nk_ohlc = idx_raw[ohlc_cols].dropna()
        else:
            nk_ohlc = pd.DataFrame(columns=ohlc_cols)  # 旧データ形式なら非表示
    else:
        mkt = pd.DataFrame(index=rel.index, columns=["TOPIX", "NIKKEI225"], dtype=float)
        nk_ohlc = pd.DataFrame(columns=ohlc_cols)

    # ウォームアップ期間を落とす
    keep = rel.index[WARMUP:]
    frames = dict(rs=rs, flow=flow, theme_flow=theme_flow, theme_rs=theme_rs)
    series = dict(rot=rot, sigma=sigma,
                  cross_up=cooldown(cross_up), cross_dn=cooldown(cross_dn),
                  plus_break=cooldown(plus_break), minus_break=cooldown(minus_break),
                  accel=cooldown(accel_alert), decel=cooldown(decel_alert))
    frames = {k: v.loc[keep] for k, v in frames.items()}
    series = {k: v.loc[keep] for k, v in series.items()}
    # 累積相対強弱は表示開始日を0%に揃える（見た目の起点を明確に）
    frames["rs"] = frames["rs"] - frames["rs"].iloc[0]
    # HTMLに埋め込むデータ量を減らすため小数を丸める（5年分×21系列対策）
    frames["flow"] = frames["flow"].round(3)
    frames["rs"] = frames["rs"].round(2)
    frames["theme_rs"] = (frames["theme_rs"] - frames["theme_rs"].iloc[0]).round(2)
    frames["theme_flow"] = frames["theme_flow"].round(3)
    frames["theme_meta"] = theme_meta
    series["rot"] = series["rot"].round(3)
    series["sigma"] = series["sigma"].round(3)

    mkt = mkt.loc[keep]
    for col in mkt.columns:
        base = mkt[col].dropna()
        if not base.empty:
            mkt[col] = mkt[col] / base.iloc[0] * 100.0
    frames["mkt_idx"] = mkt.round(2)
    frames["nk_ohlc"] = nk_ohlc[nk_ohlc.index >= keep[0]].round(1)

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

    # --- 同じグラフに日経平均・TOPIXを右軸で重ねる ---
    # 初期値は全期間の先頭=100。期間を絞ったときはHTML側のJSが
    # 「表示中の期間の先頭=100」に振り直す（rebaseIdx()）

    mkt = frames["mkt_idx"]
    idx_names = {"NIKKEI225": "日経平均", "TOPIX": "TOPIX"}
    idx_colors = {"NIKKEI225": "#eb6834", "TOPIX": "#2a78d6"}
    idx_trace_pos = []
    for col in ["NIKKEI225", "TOPIX"]:
        if col not in mkt.columns or mkt[col].dropna().empty:
            continue
        idx_trace_pos.append(len(f1.data))
        # x/y はプレーンなリストで渡す（numpy配列だとPlotlyがbase64で埋め込み、
        # 期間切替時の再計算を行うJS側から配列として読めなくなるため）
        f1.add_trace(go.Scatter(
            x=[d.strftime("%Y-%m-%d") for d in mkt.index],
            y=[None if pd.isna(v) else float(v) for v in mkt[col]],
            name=idx_names[col], yaxis="y2",
            line=dict(color=idx_colors[col], width=1.6), opacity=0.75,
            hovertemplate=idx_names[col] + "<br>%{x|%Y-%m-%d}<br>指数値: %{y:.1f}（表示期間の開始=100）<extra></extra>"))

    # 横軸はデータの両端ぴったりに固定する。自動範囲だと前後に約5%（≒3ヶ月）の
    # 余白が付き、3M/6M/1Yボタンがその余白ぶん未来側にずれて中身が消えてしまう
    # 日付のみの文字列で指定する（時刻付きだとJS側がローカル時刻として解釈し、
    # 判定境界が9時間ずれて最終日が範囲外に落ちる）
    xaxis_cfg = range_buttons()
    xaxis_cfg["range"] = [rot.index[0].strftime("%Y-%m-%d"),
                          rot.index[-1].strftime("%Y-%m-%d")]

    f1.update_layout(
        xaxis=xaxis_cfg, height=460,
        yaxis=dict(automargin=False),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, automargin=False),
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
    z = np.round(gaussian_filter(flow.to_numpy().T, sigma=(HEAT_SIGMA_B, 0), mode="nearest"), 3)
    zmax = np.percentile(np.abs(z), 98)
    f2 = go.Figure(go.Contour(
        z=z, x=flow.index, y=[short(labels[c]) for c in flow.columns],
        colorscale="RdBu", reversescale=True, zmid=0, zmin=-zmax, zmax=zmax,
        line=dict(color="rgba(30,34,45,0.55)", width=1),
        contours=dict(coloring="heatmap"),
        colorbar=dict(title=dict(text="流入<br>↑<br>↓<br>流出", font=dict(size=10)),
                      thickness=10, x=1.0, xanchor="left", tickfont=dict(size=9)),
        hovertemplate="%{y}<br>%{x|%Y-%m-%d}<br>フロー: %{z:.3f} %/日<extra></extra>"))
    f2.update_layout(xaxis=range_buttons(), height=680,
                     yaxis=dict(tickfont=dict(size=10), automargin=False))

    # --- 図1c: 日経平均ローソク足（実際の株価・円） ---
    nk = frames["nk_ohlc"]
    f1c = go.Figure()
    if not nk.empty:
        f1c.add_trace(go.Candlestick(
            x=nk.index, open=nk["NK_O"], high=nk["NK_H"], low=nk["NK_L"], close=nk["NIKKEI225"],
            name="日経平均",
            increasing=dict(line=dict(color="#d64550", width=1), fillcolor="#d64550"),
            decreasing=dict(line=dict(color="#3b6fd4", width=1), fillcolor="#3b6fd4")))
    f1c.update_layout(
        xaxis={**range_buttons(), "rangeslider": {"visible": False}},
        yaxis=dict(automargin=False, tickformat=",d"),
        height=420, showlegend=False)

    # 断面図・累積相対強弱の初期表示セクター（表示切替はHTML側のチェックボックス）
    rs = frames["rs"]
    default_visible = {c for c in rs.columns if labels[c] in DEFAULT_SECTORS}

    # --- 図2b: 資金フロー断面（地形図の各行を折れ線で抽出） ---
    f2b = go.Figure()
    f2b.add_hline(y=0, line=dict(color="#9aa0a6", width=1))
    for i, col in enumerate(flow.columns):
        f2b.add_trace(go.Scatter(
            x=flow.index, y=flow[col], name=labels[col],
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            visible=col in default_visible, showlegend=False,
            hovertemplate=labels[col] + "<br>%{x|%Y-%m-%d}<br>フロー: %{y:.3f} %/日<extra></extra>"))
    f2b.update_layout(xaxis=range_buttons(), height=440,
                      yaxis=dict(automargin=False))

    # --- 図3: 累積相対強弱 ---
    f3 = go.Figure()
    f3.add_hline(y=0, line=dict(color="#9aa0a6", width=1))
    for i, col in enumerate(rs.columns):
        # y を明示的に list にする。numpy 配列のままだと plotly が base64 で
        # 埋め込むため、期間に応じて起点を0%に振り直すJS側から読めなくなる。
        f3.add_trace(go.Scatter(
            x=rs.index, y=rs[col].tolist(), name=labels[col],
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            visible=col in default_visible, showlegend=False,
            hovertemplate=labels[col] + "<br>%{x|%Y-%m-%d}<br>市場比: %{y:.1f}%<extra></extra>"))
    f3.update_layout(xaxis=range_buttons(), height=470,
                     yaxis=dict(automargin=False))

    # --- 図4・5: テーマ層（地形図・断面図）と 図6: テーマ累積相対強弱 ---
    tflow, trs, tmeta = frames["theme_flow"], frames["theme_rs"], frames["theme_meta"]
    tcols = list(tflow.columns)
    tlab = {c: tmeta[c]["label"] for c in tcols}

    # 表示は themes.json の並び（AI・半導体…）が上から来るよう逆順にする。
    # テーマは縦の並びに意味が薄いので行方向のぼかしはセクターより弱くし、
    # 等高線の本数は明示指定する（自動だとテーマは値のばらつきが大きく線が過密になる）。
    tdisp = tcols[::-1]
    # 時間方向にも軽くぼかす。テーマは構成銘柄が少ないものがあり値の振れが大きいため、
    # 全期間表示で等高線が過密になるのを抑える（表示用のみ。数値そのものは変えない）
    tz = np.round(gaussian_filter(tflow[tdisp].to_numpy().T,
                                  sigma=(HEAT_SIGMA_B * 0.6, 3), mode="nearest"), 3)
    tzmax = np.percentile(np.abs(tz), 98)
    f4 = go.Figure(go.Contour(
        z=tz, x=tflow.index, y=[short(tlab[c]) for c in tdisp],
        colorscale="RdBu", reversescale=True, zmid=0,
        line=dict(color="rgba(30,34,45,0.5)", width=1),
        contours=dict(coloring="heatmap", showlines=True,
                      start=-tzmax, end=tzmax, size=round(tzmax / 6, 4)),
        colorbar=dict(title=dict(text="流入<br>↑<br>↓<br>流出", font=dict(size=10)),
                      thickness=10, x=1.0, xanchor="left", tickfont=dict(size=9)),
        hovertemplate="%{y}<br>%{x|%Y-%m-%d}<br>フロー: %{z:.3f} %/日<extra></extra>"))
    f4.update_layout(xaxis=range_buttons(), height=760,
                     yaxis=dict(tickfont=dict(size=10), automargin=False))

    t_default = set(trs.iloc[-1].abs().sort_values(ascending=False).index[:8])
    f5 = go.Figure()
    f5.add_hline(y=0, line=dict(color="#9aa0a6", width=1))
    for i, col in enumerate(tcols):
        f5.add_trace(go.Scatter(
            x=tflow.index, y=tflow[col], name=tlab[col],
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            visible=col in t_default, showlegend=False,
            hovertemplate=tlab[col] + "<br>%{x|%Y-%m-%d}<br>フロー: %{y:.3f} %/日<extra></extra>"))
    f5.update_layout(xaxis=range_buttons(), height=440, yaxis=dict(automargin=False))

    f6 = go.Figure()
    f6.add_hline(y=0, line=dict(color="#9aa0a6", width=1))
    for i, col in enumerate(tcols):
        f6.add_trace(go.Scatter(
            x=trs.index, y=trs[col].tolist(), name=tlab[col],
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            visible=col in t_default, showlegend=False,
            hovertemplate=tlab[col] + "<br>%{x|%Y-%m-%d}<br>市場比: %{y:.1f}%<extra></extra>"))
    f6.update_layout(xaxis=range_buttons(), height=470, yaxis=dict(automargin=False))

    for f in (f1, f1c, f2, f2b, f3, f4, f5, f6):
        f.update_layout(
            template="plotly_white", font=dict(family="'Helvetica Neue',Arial,'Hiragino Sans','Yu Gothic',sans-serif", size=12),
            margin=MARGIN,
            legend=dict(orientation="h", y=-0.12, font=dict(size=10)),
            hoverlabel=dict(font_size=12))
    return f1, f1c, f2, f2b, f3, f4, f5, f6, default_visible, t_default


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
  .selgrid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
              gap: 2px 8px; font-size: .8rem; padding: 6px 10px 10px; }}
  .selgrid label {{ display: flex; align-items: center; gap: 5px; cursor: pointer;
                    padding: 3px 2px; white-space: nowrap; }}
  .selgrid input {{ accent-color: #2a78d6; width: 15px; height: 15px; flex: none; }}
  .chip {{ display: inline-block; width: 11px; height: 11px; border-radius: 3px; flex: none; }}
  .selbtns {{ padding: 4px 10px 0; }}
  .selbtns button {{ font-size: .8rem; padding: 5px 12px; margin-right: 8px; cursor: pointer;
                     border: 1px solid #d1d5db; border-radius: 6px; background: #f9fafb; }}
  .selbtns button:active {{ background: #e5e7eb; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>日経225 セクターローテーション</h1>
  <div class="updated">最終更新: {updated}（毎営業日 夜に自動更新）</div>
  <details>
    <summary>この指標の読み方</summary>
    <ul>
      <li><b>ローテーション指数</b>: 攻めのバスケット（半導体AI・銀行・商社など17分類）と守りのバスケット（電力ガス鉄道・食品・医薬品・通信）の資金フローの差。プラス圏＝リスクオン、マイナス圏＝リスクオフ。</li>
      <li><b>マーカーの意味</b>:
        <ul style="margin:4px 0; padding-left:1.2em">
          <li><span style="color:#2f9e44">●</span> <b>ゼロクロス↑</b>: 指数がマイナス→プラスに転換。守りから攻めへ資金が動き始めたサイン</li>
          <li><span style="color:#d64550">●</span> <b>ゼロクロス↓</b>: プラス→マイナスに転換。攻めから守りへ退避が始まったサイン</li>
          <li><span style="color:#3b6fd4">●</span> <b>+σブレイク</b>: リスクオンの偏りが普段の変動幅（±σ・灰色点線）を上抜け。過熱気味の警戒サイン</li>
          <li><span style="color:#8a3ffc">●</span> <b>−σブレイク</b>: リスクオフ方向に行き過ぎ。悲観が極まっている（反発が近いことも）</li>
          <li><span style="color:#2f9e44">▲</span> <b>加速アラート</b>: 指数の勢いが急に強まった（流れの初動を示唆）</li>
          <li><span style="color:#d64550">▼</span> <b>減速アラート</b>: 勢いが急に衰えた（トレンド失速の初動を示唆）</li>
        </ul>
      </li>
      <li><b>日経平均・TOPIX（右軸）</b>: 実際の相場の値動き。3M/6M/1Y/Allを切り替えると、表示中の期間の先頭が100になるよう自動で振り直されます（その期間で何%動いたかがそのまま読めます）。ローテーション指数がプラスなのに相場が下げている、といったズレを確認できます。右上のボタンでまとめて表示/非表示を切り替えられます。</li>
      <li><b>資金フロー地形図</b>: 各バスケットへの資金の流入（赤）・流出（青）の強さ。縦に見ると「今どこが買われているか」、横に見ると「そのテーマがいつから続いているか」。</li>
      <li><b>資金フロー断面図</b>: 地形図を横から見た図。各線は地形図の1行と同じデータで、線がプラス圏＝流入（地形図の赤）、マイナス圏＝流出（青）に対応します。</li>
      <li><b>累積相対強弱</b>: 「市場平均」（全24セクターの平均リターン）に対する超過リターンの積み上げ。<b>表示中の期間の先頭を0%</b>として、そこから市場にどれだけ勝った/負けたかを表します（3M/6M/1Y/Allを切り替えると起点も自動で切り替わります）。右肩上がり＝市場より強い。資金フロー（地形図・断面図）はこのグラフの「傾き」にあたり、3つのグラフは同じ計算のつながりで対応しています。</li>
      <li><b>投資テーマ</b>（ページ下部）: みんかぶ・株探の人気テーマを参考にした分類。セクターと違い<b>銘柄が重複してよく、日経225以外の銘柄も含みます</b>（SaaS・外食など）。テーマはローテーション指数や市場平均の計算には一切使わない表示専用です。</li>
      <li><b>シグナルは一度出たら動きません</b>: 平滑化にその日までのデータしか使わない計算方式のため、後日データが増えても過去のシグナル位置は変わりません（2026-07-18に変更。それ以前は前後の日を平均する方式で、表示済みのシグナルが後から移動していました）。代わりに反応は数日遅れます。</li>
      <li>数値はすべて株価から計算した加工済みの独自指標です。投資判断はご自身の責任で行ってください。</li>
    </ul>
  </details>
  <div class="card"><h2>ローテーション指数（＋＝リスクオン ／ −＝リスクオフ）</h2>
    <p class="note">細い線は日経平均・TOPIX（右軸・表示中の期間の先頭を100に自動調整）。右上のボタンでまとめて表示/非表示</p>{fig1}</div>
  <div class="card"><h2>日経平均（ローソク足・円）</h2>
    <p class="note">赤＝陽線（終値が始値より高い）、青＝陰線。上のローテーション指数と同じ横軸なので、シグナルと値動きを縦に見比べられます</p>{fig1c}</div>
  <div class="card"><h2>資金フロー地形図（赤＝流入・青＝流出）</h2>{fig2}</div>
  <div class="card">
    <h2>表示セクターの選択</h2>
    <p class="note">下2つのグラフ（断面図・累積相対強弱）に反映されます</p>
    <div class="selbtns">
      <button type="button" onclick="allSectors(true)">全て表示</button>
      <button type="button" onclick="allSectors(false)">全て非表示</button>
    </div>
    <div class="selgrid" id="selgrid">{sector_checkboxes}</div>
  </div>
  <div class="card"><h2>資金フロー断面図（地形図を横から見た図・%/日）</h2>
    <p class="note">各線＝地形図の1行。プラス圏＝流入、マイナス圏＝流出。表示は上の選択パネルで切り替え</p>{fig2b}</div>
  <div class="card"><h2>累積相対強弱（市場平均比）</h2>
    <p class="note">右肩上がり＝市場平均より強い。期間を切り替えると、その期間の先頭が0%になるよう自動で振り直されます</p>{fig3}</div>
  <h2 style="margin:26px 0 4px">投資テーマ（みんかぶ・株探の人気テーマより）</h2>
  <p class="note" style="margin:0 0 10px">
    セクターと違い銘柄が重複し、日経225以外の銘柄も含みます。上のローテーション指数の計算には使っていません</p>
  <div class="card">
    <h2>表示テーマの選択</h2>
    <p class="note">下2つのグラフに反映されます。初期表示は直近の動きが大きい8テーマ</p>
    <div class="selbtns">
      <button type="button" onclick="allThemes(true)">全て表示</button>
      <button type="button" onclick="allThemes(false)">全て非表示</button>
    </div>
    <div class="selgrid" id="thgrid">{theme_checkboxes}</div>
  </div>
  <div class="card"><h2>テーマ別 資金フロー地形図</h2>{fig4}</div>
  <div class="card"><h2>テーマ別 資金フロー断面図（%/日）</h2>{fig5}</div>
  <div class="card"><h2>テーマ別 累積相対強弱（市場平均比）</h2>
    <p class="note">期間を切り替えると、その期間の先頭が0%になるよう自動で振り直されます</p>{fig6}</div>
  <details>
    <summary>各セクターの採用銘柄一覧</summary>
    {baskets_html}
  </details>
  <details>
    <summary>各テーマの採用銘柄一覧</summary>
    {themes_html}
  </details>
</div>
<script>
// --- 期間に合わせて基準を振り直す ---
// 期間ボタンやズームは横軸を変えるだけなので、表示中の期間の先頭が
//   ・日経平均/TOPIX（右軸） → 100
//   ・累積相対強弱（セクター・テーマ）→ 0%
// になるようJS側で毎回計算し直す。
// 注意: StatiCrypt はパスワード入力後にページ内容を差し込むため load イベントは
// すでに終わっている。load を待たずポーリングで初期化すること。
function pt(v) {{ return typeof v === "number" ? v : new Date(v).getTime(); }}

function grabRaw(gd, filter) {{
  // plotly は数値配列を base64 で埋め込むことがあるので _inputArray も見る
  var st = {{gd: gd, traces: [], y: [], t: [], clamping: false}};
  gd.data.forEach(function (tr, i) {{
    if (!filter(tr)) return;
    var y = tr.y;
    if (y && !Array.isArray(y)) y = y._inputArray || y;
    if (!y || typeof y.length !== "number" || !y.length) return;
    st.traces.push(i);
    st.y.push(Array.prototype.slice.call(y));
    st.t.push(Array.prototype.map.call(tr.x, function (d) {{
      return new Date(d).getTime();
    }}));
  }});
  if (!st.traces.length) return null;
  var flat = [].concat.apply([], st.t);
  st.tmin = Math.min.apply(null, flat);
  st.tmax = Math.max.apply(null, flat);
  return st;
}}

// 期間ボタンは横軸の右端から遡って範囲を決めるため、右端がデータ最終日より
// 未来にあると表示がずれる（3Mだと中身が空になる）。はみ出しを検出したら
// データ最終日で終わるよう同じ幅で引き戻す。
function clampX(st, after) {{
  if (st.clamping) return false;
  var xa = st.gd.layout.xaxis;
  if (!xa || !xa.range) return false;
  var lo = pt(xa.range[0]), hi = pt(xa.range[1]);
  if (hi <= st.tmax + 864e5) return false;
  var nlo = Math.max(st.tmin, st.tmax - (hi - lo));
  st.clamping = true;
  Plotly.relayout(st.gd, {{"xaxis.range": [new Date(nlo).toISOString(),
                                          new Date(st.tmax).toISOString()]}})
        .then(function () {{ st.clamping = false; after(); }});
  return true;
}}

function makeRebase(st, mode) {{
  return function () {{
    var xa = st.gd.layout.xaxis, lo = -Infinity, hi = Infinity;
    if (xa && xa.range) {{ lo = pt(xa.range[0]); hi = pt(xa.range[1]); }}
    var ys = st.y.map(function (raw, k) {{
      var ts = st.t[k], base = null;
      return raw.map(function (v, i) {{
        // 表示範囲外は null にして、縦軸が表示中のデータに合わせて自動調整されるようにする
        if (v == null || isNaN(v) || ts[i] < lo || ts[i] > hi) return null;
        if (base === null) base = v;
        return mode === "index"
          ? Math.round(v / base * 1000) / 10       // 先頭=100
          : Math.round((v - base) * 100) / 100;    // 先頭=0%
      }});
    }});
    Plotly.restyle(st.gd, {{y: ys}}, st.traces);
  }};
}}

function setupRebase(id, filter, mode) {{
  var gd = document.getElementById(id);
  if (!gd || !gd.data) return false;
  var st = grabRaw(gd, filter);
  if (!st) return false;
  var fn = makeRebase(st, mode);
  fn();
  gd.on("plotly_relayout", function () {{ if (!clampX(st, fn)) fn(); }});
  return true;
}}

(function initRebase(tries) {{
  var targets = [
    ["figrot", function (t) {{ return t.yaxis === "y2"; }}, "index"],
    ["figrs",  function (t) {{ return true; }}, "zero"],
    ["figtrs", function (t) {{ return true; }}, "zero"]
  ];
  var ready = targets.every(function (a) {{
    var g = document.getElementById(a[0]);
    return g && g.data && g.data.length;
  }});
  if (!ready) {{
    if (tries < 150) setTimeout(function () {{ initRebase(tries + 1); }}, 200);
    return;
  }}
  targets.forEach(function (a) {{ setupRebase(a[0], a[1], a[2]); }});
}})(0);

function selFigs() {{
  return [document.getElementById("figflow"), document.getElementById("figrs")];
}}
function oneSector(cb) {{
  const i = Number(cb.dataset.i);
  selFigs().forEach(gd => Plotly.restyle(gd, {{visible: cb.checked}}, [i]));
}}
function allSectors(on) {{
  const boxes = document.querySelectorAll("#selgrid input");
  boxes.forEach(cb => {{ cb.checked = on; }});
  const idx = Array.from(boxes, cb => Number(cb.dataset.i));
  selFigs().forEach(gd => Plotly.restyle(gd, {{visible: on}}, idx));
}}
// --- テーマ層の表示切替（断面図と累積相対強弱に反映） ---
function themeFigs() {{
  return [document.getElementById("figtflow"), document.getElementById("figtrs")];
}}
function oneTheme(cb) {{
  const i = Number(cb.dataset.i);
  themeFigs().forEach(gd => Plotly.restyle(gd, {{visible: cb.checked}}, [i]));
}}
function allThemes(on) {{
  const boxes = document.querySelectorAll("#thgrid input");
  boxes.forEach(cb => {{ cb.checked = on; }});
  const idx = Array.from(boxes, cb => Number(cb.dataset.i));
  themeFigs().forEach(gd => Plotly.restyle(gd, {{visible: on}}, idx));
}}
</script>
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


def themes_table_html(theme_meta) -> str:
    """テーマ別の採用銘柄一覧（銘柄名・コードのみ＝公開情報）"""
    cfg = json.loads((ROOT / "themes.json").read_text(encoding="utf-8"))
    bk = json.loads((ROOT / "baskets.json").read_text(encoding="utf-8"))
    parts, cur_group = [], None
    for key, t in cfg["themes"].items():
        if key not in theme_meta:
            continue
        group = t.get("group", "その他")
        if group != cur_group:
            cur_group = group
            parts.append(f'<p style="margin:12px 0 2px;color:#6b7280">― {group} ―</p>')
        if "from_baskets" in t:
            names = "、".join(bk["baskets"][b]["label"] for b in t["from_baskets"])
            body = f'（基本層の合成: {names}）'
        else:
            body = "、".join(f"{co}({code})" for code, co in t["codes"].items())
        parts.append(f'<p style="margin:6px 0"><b>{t["label"]}</b>'
                     f'<span style="color:#6b7280">（{theme_meta[key]["n"]}銘柄）</span><br>{body}</p>')
    return "\n".join(parts)


def main() -> None:
    if not PRICES.exists():
        sys.exit("data/prices.parquet がありません。先に fetch_data.py を実行してください。")
    frames, series, labels = compute()
    (f1, f1c, f2, f2b, f3, f4, f5, f6,
     default_visible, t_default) = build_figs(frames, series, labels)

    cfg = {"responsive": True, "displaylogo": False,
           "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}
    kw = dict(full_html=False, include_plotlyjs=False, config=cfg)
    parts = [f1.to_html(div_id="figrot", **kw), f1c.to_html(**kw), f2.to_html(**kw),
             f2b.to_html(div_id="figflow", **kw), f3.to_html(div_id="figrs", **kw),
             f4.to_html(**kw), f5.to_html(div_id="figtflow", **kw),
             f6.to_html(div_id="figtrs", **kw)]

    def checkboxes(cols, label_of, visible, handler):
        """グラフのトレース順と同じ並びのチェックボックス群"""
        out = []
        for i, col in enumerate(cols):
            checked = " checked" if col in visible else ""
            out.append(
                f'<label><input type="checkbox" data-i="{i}"{checked} onchange="{handler}(this)">'
                f'<span class="chip" style="background:{PALETTE[i % len(PALETTE)]}"></span>'
                f'{label_of(col)}</label>')
        return "\n".join(out)

    tmeta = frames["theme_meta"]
    jst = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=9)
    html = HTML_TEMPLATE.format(
        updated=jst.strftime("%Y-%m-%d %H:%M JST"),
        fig1=parts[0], fig1c=parts[1], fig2=parts[2], fig2b=parts[3], fig3=parts[4],
        fig4=parts[5], fig5=parts[6], fig6=parts[7],
        sector_checkboxes=checkboxes(frames["flow"].columns, lambda c: labels[c],
                                     default_visible, "oneSector"),
        theme_checkboxes=checkboxes(frames["theme_flow"].columns,
                                    lambda c: tmeta[c]["label"], t_default, "oneTheme"),
        baskets_html=baskets_table_html(),
        themes_html=themes_table_html(tmeta))
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"出力完了: {OUT_DIR / 'index.html'}（データ最終日: {series['rot'].index[-1]:%Y-%m-%d}）")


if __name__ == "__main__":
    main()
