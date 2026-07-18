# -*- coding: utf-8 -*-
"""
============================================================
 バスケット定義の生成（日経225全銘柄 → 21セクター）
============================================================
 data/nikkei225_members.json（日経公式サイトから取得した採用銘柄と
 日経業種分類）を元に、テーマ性を加味した21バスケットの
 baskets.json を生成します。

 ・日経業種 → バスケットの既定マッピング＋銘柄単位の上書き（OVERRIDE）
 ・日経平均の銘柄入れ替え時は fetch_members.py → このスクリプトを再実行
============================================================
"""
from pathlib import Path
import json
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
MEMBERS = ROOT / "data" / "nikkei225_members.json"

# バスケット定義（この順序がヒートマップの並び。先頭=下段・守り → 末尾=上段・攻め）
LABELS = {
    "infra_defensive":        "電力・ガス・鉄道",
    "food_household":         "食品・生活用品",
    "pharma":                 "医薬品",
    "telecom":                "通信",
    "transport":              "運輸・物流",
    "construction_realestate": "建設・不動産・住宅",
    "retail":                 "小売",
    "financials":             "保険・証券・その他金融",
    "banks":                  "銀行",
    "trading_houses":         "商社",
    "energy":                 "資源・エネルギー",
    "chemicals":              "化学・素材",
    "steel_nonferrous":       "鉄鋼・非鉄・電線",
    "auto":                   "自動車",
    "internet_services":      "ネット・ITサービス",
    "games_entertainment":    "ゲーム・エンタメ",
    "machinery_fa":           "機械・FA・ロボット",
    "precision_medical":      "精密・医療機器",
    "electronics":            "電機・電子部品",
    "heavy_defense":          "重工・防衛",
    "semiconductor_ai":       "半導体・AI",
}

RISK_OFF = ["infra_defensive", "food_household", "pharma", "telecom"]

# 日経業種 → バスケットの既定マッピング
SECTOR_MAP = {
    "医薬品": "pharma",
    "電気機器": "electronics",
    "自動車": "auto",
    "精密機器": "precision_medical",
    "通信": "telecom",
    "銀行": "banks",
    "その他金融": "financials",
    "証券": "financials",
    "保険": "financials",
    "水産": "food_household",
    "食品": "food_household",
    "小売業": "retail",
    "サービス": "internet_services",
    "鉱業": "energy",
    "石油": "energy",
    "繊維": "chemicals",
    "パルプ・紙": "chemicals",
    "化学": "chemicals",
    "窯業": "chemicals",
    "ゴム": "auto",              # 採用2銘柄はいずれもタイヤ
    "鉄鋼": "steel_nonferrous",
    "非鉄・金属": "steel_nonferrous",
    "商社": "trading_houses",
    "建設": "construction_realestate",
    "不動産": "construction_realestate",
    "機械": "machinery_fa",
    "造船": "heavy_defense",     # 川崎重工
    "その他製造": "electronics",  # 個別上書きで振り分け
    "鉄道・バス": "infra_defensive",
    "陸運": "transport",
    "海運": "transport",
    "空運": "transport",
    "電力": "infra_defensive",
    "ガス": "infra_defensive",
}

# 銘柄単位の上書き（テーマ性を優先）
OVERRIDE = {
    # 半導体・AI（業種を跨いで集約）
    "8035": "semiconductor_ai",  # 東京エレクトロン
    "6857": "semiconductor_ai",  # アドバンテスト
    "6146": "semiconductor_ai",  # ディスコ（精密機器から）
    "6920": "semiconductor_ai",  # レーザーテック
    "7735": "semiconductor_ai",  # SCREEN
    "6723": "semiconductor_ai",  # ルネサス
    "285A": "semiconductor_ai",  # キオクシア
    "6526": "semiconductor_ai",  # ソシオネクスト
    "4062": "semiconductor_ai",  # イビデン
    "3436": "semiconductor_ai",  # SUMCO（非鉄から・シリコンウエハ）
    "6963": "semiconductor_ai",  # ローム（パワー半導体）
    "9984": "semiconductor_ai",  # ソフトバンクG（AI投資会社）
    # 機械・FA・ロボット（電気機器のFA勢を移動）
    "6954": "machinery_fa",      # ファナック
    "6506": "machinery_fa",      # 安川電機
    "6861": "machinery_fa",      # キーエンス
    "6645": "machinery_fa",      # オムロン
    "6841": "machinery_fa",      # 横河電機
    "6479": "machinery_fa",      # ミネベアミツミ（ベアリング）
    # 重工・防衛
    "7011": "heavy_defense",     # 三菱重工（機械から）
    "7013": "heavy_defense",     # IHI（機械から）
    "5631": "heavy_defense",     # 日本製鋼所（防衛・砲身）
    # 自動車部品・タイヤ
    "6902": "auto",              # デンソー（電気機器から）
    # 精密機器のうちOA機器は電機へ
    "4902": "electronics",       # コニカミノルタ
    # 生活用品（化学から）
    "4452": "food_household",    # 花王
    "4911": "food_household",    # 資生堂
    # 電子材料
    "6988": "electronics",       # 日東電工（化学から）
    # 住宅設備
    "5332": "construction_realestate",  # TOTO（窯業から）
    # ゲーム・エンタメ（サービス・その他製造から）
    "7974": "games_entertainment",  # 任天堂
    "3659": "games_entertainment",  # ネクソン
    "2432": "games_entertainment",  # DeNA
    "9766": "games_entertainment",  # コナミG
    "9602": "games_entertainment",  # 東宝
    "4661": "games_entertainment",  # オリエンタルランド
    "7832": "games_entertainment",  # バンダイナムコ
    "7951": "games_entertainment",  # ヤマハ（楽器）
    # ヘルスケア
    "2413": "pharma",            # エムスリー（医療サービス）
    # 金融
    "6178": "financials",        # 日本郵政
}


def main() -> None:
    members = json.loads(MEMBERS.read_text(encoding="utf-8"))
    assert len(members) == 225, f"採用銘柄数が225ではありません: {len(members)}"

    baskets: dict[str, dict] = {k: {"label": v, "codes": {}} for k, v in LABELS.items()}
    for m in members:
        code, name, sector = m["code"], m["name"], m["nikkei_sector"]
        basket = OVERRIDE.get(code) or SECTOR_MAP.get(sector)
        if basket is None:
            sys.exit(f"マッピング未定義: {code} {name}（{sector}）")
        baskets[basket]["codes"][code] = name

    total = sum(len(b["codes"]) for b in baskets.values())
    assert total == 225, f"振り分け後の合計が225ではありません: {total}"

    out = {
        "comment": "日経225全銘柄を21バスケットに分類。make_baskets.py で生成（手編集も可）。リスト順がヒートマップの並び（先頭=下段・守り）。",
        "risk_off": RISK_OFF,
        "baskets": baskets,
    }
    (ROOT / "baskets.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for k, b in baskets.items():
        print(f"{LABELS[k]}: {len(b['codes'])}銘柄")
    print(f"合計 {total} 銘柄 / {len(baskets)} バスケット → baskets.json 更新完了")


if __name__ == "__main__":
    main()
