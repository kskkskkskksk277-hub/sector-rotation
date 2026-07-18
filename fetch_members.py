# -*- coding: utf-8 -*-
"""
============================================================
 日経225採用銘柄リストの取得
============================================================
 日経公式サイトから採用銘柄（コード・銘柄名・日経業種）を取得し、
 data/nikkei225_members.json に保存します。

 日経平均の銘柄入れ替え（定期見直しは4月・10月）の後に実行し、
 続けて make_baskets.py で baskets.json を作り直してください。
============================================================
"""
from pathlib import Path
import json
import re
import sys

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "nikkei225_members.json"
URL = "https://indexes.nikkei.co.jp/nkave/index/component?idx=nk225"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def main() -> None:
    r = requests.get(URL, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    html = r.text

    pat = re.compile(
        r'<h3[^>]*>([^<]+)</h3>'
        r'|<td>([0-9][0-9A-Z]{3})</td>\s*<td><a[^>]*>([^<]+)</a></td>\s*<td>([^<]+)</td>')
    sector = None
    out = []
    for m in pat.finditer(html):
        if m.group(1):
            sector = m.group(1).strip()
        else:
            out.append({"code": m.group(2), "name": m.group(3).strip(),
                        "full": m.group(4).strip(), "nikkei_sector": sector})

    if len(out) != 225:
        sys.exit(f"抽出数が225ではありません: {len(out)}（サイト構造が変わった可能性）")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"保存完了: {OUT}（{len(out)} 銘柄）")


if __name__ == "__main__":
    main()
