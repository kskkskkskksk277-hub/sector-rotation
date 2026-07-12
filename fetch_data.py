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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

API_BASE = "https://api.jquants.com/v2"
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PRICES = DATA_DIR / "prices.parquet"

# 表示1.5年 + 指標計算のウォームアップ分を遡る（暦日）
LOOKBACK_DAYS = 820


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
        time.sleep(0.2)  # 行儀よく

    if missing:
        log(f"注意: データが取れなかった銘柄: {missing}")
    if not frames:
        sys.exit("データが1件も取得できませんでした。APIキーを確認してください。")

    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"])
    DATA_DIR.mkdir(exist_ok=True)
    out.to_parquet(PRICES, index=False)
    log(f"保存完了: {PRICES}（{out['Code'].nunique()} 銘柄 × 約 {out['Date'].nunique()} 営業日）")


if __name__ == "__main__":
    main()
