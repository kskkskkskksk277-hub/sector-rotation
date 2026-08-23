# -*- coding: utf-8 -*-
"""
============================================================
 テーマ定義の総点検（株探のテーマ別銘柄一覧と突き合わせ）
============================================================
 themes.json の各テーマについて、株探 kabutan.jp のテーマページに
 実際に登録されている銘柄を取得し、現在の定義との差分を出力する。

 ・自分の推測ではなく実サイトの登録銘柄を根拠にするための点検スクリプト
 ・出力の「サイトにあるが未採用」から追加候補を、
   「採用しているがサイトに無い」から除外候補を判断する
============================================================
"""
from pathlib import Path
import json
import re
import sys
import time

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

# themes.json のキー → 株探のテーマ名（複数指定した場合は合算）
KABUTAN_THEMES = {
    "datacenter":       ["データセンター"],
    "semi_equipment":   ["半導体製造装置"],
    "semi_materials":   ["半導体部材・部品"],
    "optical_comm":     ["光デバイス"],
    "saas":             ["SaaS"],
    "ai_software":      ["AIエージェント"],
    "cyber_security":   ["サイバーセキュリティ"],
    "quantum":          ["量子コンピューター"],
    "physical_ai":      ["フィジカルAI", "ロボット"],
    "server_cooling":   ["サーバー冷却"],
    "rare_earth":       ["レアアース"],
    "battery":          ["蓄電池"],
    "fusion":           ["核融合発電"],
    "renewable":        ["再生可能エネルギー"],
    "perovskite":       ["ペロブスカイト太陽電池"],
    "shipping_theme":   ["海運"],
    "regional_banks":   ["地方銀行"],
    "rate_hike":        ["金利上昇メリット"],
    "domestic_yen":     ["円高メリット"],
    "defense_space":    ["防衛", "宇宙開発関連"],
    "sdv":              ["SDV"],
    "restaurant":       ["外食"],
    "biotech":          ["バイオテクノロジー関連"],
    "crypto":           ["仮想通貨"],
    "drone":            ["ドローン"],
}

ROW = re.compile(
    r'<td class="tac"><a href="/stock/\?code=([0-9A-Z]{4})">[0-9A-Z]{4}</a></td>\s*'
    r'<td class="tal">([^<]+)</td>')


def fetch_theme(name: str) -> dict[str, str]:
    """株探のテーマページから {コード: 銘柄名} を取得（1ページ15件のページ送りを全部たどる）"""
    out: dict[str, str] = {}
    for page in range(1, 40):                     # 上限40ページ＝600銘柄
        r = requests.get("https://kabutan.jp/themes/",
                         params={"theme": name, "page": page},
                         headers={"User-Agent": UA}, timeout=60)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        rows = {c: n.strip() for c, n in ROW.findall(r.text)}
        new = {c: n for c, n in rows.items() if c not in out}
        if not new:                               # 同じ内容が返ったら終端
            break
        out.update(new)
        time.sleep(0.7)
    return out


def main() -> None:
    themes = json.loads((ROOT / "themes.json").read_text(encoding="utf-8"))["themes"]
    master = {}
    for rec in json.loads((ROOT / "data" / "master.json").read_text(encoding="utf-8")):
        c = rec["Code"]
        master[c[:-1] if len(c) == 5 and c.endswith("0") else c] = rec["CoName"]

    n225 = set()
    bk = json.loads((ROOT / "baskets.json").read_text(encoding="utf-8"))
    for b in bk["baskets"].values():
        n225 |= set(b["codes"])

    for key, site_names in KABUTAN_THEMES.items():
        if key not in themes:
            continue
        site: dict[str, str] = {}
        for nm in site_names:
            try:
                site.update(fetch_theme(nm))
            except Exception as e:                       # noqa: BLE001
                print(f"  ！取得失敗 {nm}: {e}")
            time.sleep(1.0)

        mine = set(themes[key].get("codes", {}))
        label = themes[key]["label"]
        print(f"\n{'='*70}\n■ {label}（株探: {' + '.join(site_names)} / 登録{len(site)}銘柄）")

        missing = [c for c in site if c not in mine]
        extra = [c for c in mine if c not in site]

        # 追加候補は日経225採用を優先して表示（値動きが大きく指標として使いやすい）
        m225 = [c for c in missing if c in n225]
        mother = [c for c in missing if c not in n225]
        if m225:
            print(f"  ◆サイトにあるが未採用【日経225銘柄】{len(m225)}件")
            print("    " + "、".join(f"{site[c]}({c})" for c in m225))
        if mother:
            print(f"  ◇サイトにあるが未採用【225外】{len(mother)}件")
            print("    " + "、".join(f"{site[c]}({c})" for c in mother[:40]))
        if extra:
            print(f"  ×採用しているがサイトに無い {len(extra)}件")
            print("    " + "、".join(
                f"{themes[key]['codes'][c]}({c})" for c in extra))
        if not missing and not extra:
            print("  → 完全一致")


if __name__ == "__main__":
    main()
