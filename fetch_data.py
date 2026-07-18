# -*- coding: utf-8 -*-
"""
============================================================
 セクターローテーション：株価データ取得
============================================================
 baskets.json に定義した全銘柄の日足（調整後終値）を
 J-Quants API V2 から銘柄ごとに取得し、data/prices.parquet に保存します。

 ・毎回全期間（約2年分）を取り直すシンプル設計（銘柄数≒90なので数分で終わる）
 ・GitHub Actions 上でもローカルでも同じように動きます

 必要なもの: 環境変数 or .env に  JQUANTS_API_KEY=（キー）
============================================================
"""
from pathlib import Path
import datetime as dt
import json
import os
import sys
import time

import pandas as pd
import requests
import yfinance as yf

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

API_BASE = "https://api.jquants.com/v2"
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PRICES = DATA_DIR / "prices.parquet"
INDICES = DATA_DIR / "indices.parquet"

# 取得期間（暦日）。J-Quants Lightプランの上限は過去5年ちょうどで、
# それより古い日付を from に指定すると 400 エラーになるため、少し内側の1800日にする
LOOKBACK_DAYS = 1800


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_api_key() -> str:
    key = os.getenv("JQUANTS_API_KEY", "").strip()
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == "JQUANTS_API_KEY":
                return v.strip().strip('"').strip("'")
    sys.exit("JQUANTS_API_KEY が設定されていません（環境変数 or .env）")


def load_codes() -> list[str]:
    cfg = json.loads((ROOT / "baskets.json").read_text(encoding="utf-8"))
    codes: list[str] = []
    for basket in cfg["baskets"].values():
        codes.extend(basket["codes"].keys())
    # J-Quants は5桁コード（4桁 + 末尾0）
    return [c + "0" if len(c) == 4 else c for c in dict.fromkeys(codes)]


def fetch_code(s: requests.Session, code: str, frm: str, to: str) -> list[dict]:
    params = {"code": code, "from": frm, "to": to}
    rows: list[dict] = []
    while True:
        r = s.get(API_BASE + "/equities/bars/daily", params=params, timeout=120)
        if r.status_code == 429:
            log("レート制限。10秒待機…")
            time.sleep(10)
            continue
        r.raise_for_status()
        body = r.json()
        rows.extend(body.get("data", []))
        pk = body.get("pagination_key")
        if not pk:
            return rows
        params["pagination_key"] = pk


def fetch_topix(s: requests.Session, frm: str, to: str) -> pd.DataFrame:
    """TOPIX: J-Quants専用エンドポイント（Lightプランで利用可）"""
    params = {"from": frm, "to": to}
    rows: list[dict] = []
    while True:
        r = s.get(API_BASE + "/indices/bars/daily/topix", params=params, timeout=120)
        r.raise_for_status()
        body = r.json()
        rows.extend(body.get("data", []))
        pk = body.get("pagination_key")
        if not pk:
            break
        params["pagination_key"] = pk
    df = pd.DataFrame(rows)[["Date", "C"]].rename(columns={"C": "TOPIX"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def fetch_nikkei225(frm: str, to: str) -> pd.DataFrame:
    """日経平均: J-Quants Lightプランでは非対応（403）のため yfinance で代替"""
    df = yf.download("^N225", start=frm, end=to, progress=False, auto_adjust=False)
    if df.empty:
        log("警告: 日経平均の取得に失敗しました（yfinance）。指数比較チャートから除外します。")
        return pd.DataFrame(columns=["Date", "NIKKEI225"])
    close = df["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    out = close.reset_index()
    out.columns = ["Date", "NIKKEI225"]
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    return out


def update_indices(frm_dt: dt.date, to_dt: dt.date, key: str) -> None:
    s = requests.Session()
    s.headers["x-api-key"] = key
    frm, to = frm_dt.strftime("%Y%m%d"), to_dt.strftime("%Y%m%d")
    log("TOPIXを取得します…")
    topix = fetch_topix(s, frm, to)
    log("日経平均を取得します（yfinance）…")
    nikkei = fetch_nikkei225(frm_dt.isoformat(), (to_dt + dt.timedelta(days=1)).isoformat())
    merged = topix.merge(nikkei, on="Date", how="outer").sort_values("Date")
    merged.to_parquet(INDICES, index=False)
    log(f"指数データ保存完了: {INDICES}（{len(merged)} 営業日）")


def main() -> None:
    key = load_api_key()
    s = requests.Session()
    s.headers["x-api-key"] = key

    codes = load_codes()
    today = dt.date.today()
    frm = (today - dt.timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    to = today.strftime("%Y%m%d")
    log(f"{len(codes)} 銘柄の日足を取得します（{frm} 〜 {to}）")

    frames = []
    missing = []
    for i, code in enumerate(codes, 1):
        rows = fetch_code(s, code, frm, to)
        if rows:
            df = pd.DataFrame(rows)[["Date", "Code", "AdjC"]]
            frames.append(df)
        else:
            missing.append(code)
        if i % 20 == 0:
            log(f"  {i}/{len(codes)} 銘柄完了")
        time.sleep(0.05)  # 行儀よく（レート制限時は自動で10秒待機）

    if missing:
        log(f"注意: データが取れなかった銘柄: {missing}")
    if not frames:
        sys.exit("データが1件も取得できませんでした。APIキーを確認してください。")

    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"])
    DATA_DIR.mkdir(exist_ok=True)
    out.to_parquet(PRICES, index=False)
    log(f"保存完了: {PRICES}（{out['Code'].nunique()} 銘柄 × 約 {out['Date'].nunique()} 営業日）")

    update_indices(today - dt.timedelta(days=LOOKBACK_DAYS), today, key)


if __name__ == "__main__":
    main()
