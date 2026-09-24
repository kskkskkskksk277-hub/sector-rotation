# -*- coding: utf-8 -*-
"""
ローテーション指数の −0.5 割れを Discord に通知する（Webhook）

 ・当日の指数が −0.5 を下回った（前日は −0.5 以上）→ 買いシグナル通知
 ・シグナルから 20 営業日後 → 利益確定の目安日として通知
 ・検証根拠: research/threshold_buy.py / threshold_buy_etf17.py（20日後に優位、60日後は優位が消える）

 環境変数: DISCORD_WEBHOOK_URL。未設定なら何もせず終了する（ダッシュボード更新は止めない）。
 `python notify_discord.py --test` で現在値をテスト送信する。
"""
import json
import os
import sys
import urllib.request

from build_dashboard import compute

THRESHOLD = -0.5
HOLD_DAYS = 20
STOP_LOSS = -8  # %


def push(text: str) -> None:
    url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not url:
        print("DISCORD_WEBHOOK_URL が無いため通知をスキップ")
        return
    body = json.dumps({"content": text}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": "sector-rotation"})
    with urllib.request.urlopen(req, timeout=20) as res:
        print("Discord 送信:", res.status)


def main() -> None:
    _, series, _ = compute()
    rot = series["rot"]
    today, value, prev = rot.index[-1], rot.iloc[-1], rot.iloc[-2]
    date = f"{today:%Y/%m/%d}"
    crossed = (rot < THRESHOLD) & (rot.shift() >= THRESHOLD)

    if "--test" in sys.argv:
        push(f"✅ テスト送信\n{date} のローテーション指数: {value:+.2f}\n"
             f"（{THRESHOLD} を下回った日に買いシグナルを通知します）")
        return

    if crossed.iloc[-1]:
        push(f"🔄 **セクターローテ 買いシグナル**（{date}）\n"
             f"ローテーション指数 {value:+.2f}（{THRESHOLD} 割れ・前日 {prev:+.2f}）\n\n"
             f"・明朝の寄付きで日経連動ETF（1321等）を買う\n"
             f"・{HOLD_DAYS}営業日後に利益確定（その日も通知します）\n"
             f"・買値から{STOP_LOSS}%で損切り\n\n"
             f"過去実績: 20日後平均 +3.7〜6.7%（勝率70〜100%）。暴落途中で買うため損切り厳守")
    if len(crossed) > HOLD_DAYS and crossed.iloc[-1 - HOLD_DAYS]:
        sig_day = rot.index[-1 - HOLD_DAYS]
        push(f"🔄 **セクターローテ 利益確定の目安日**（{date}）\n"
             f"{sig_day:%Y/%m/%d} の買いシグナルから{HOLD_DAYS}営業日が経過しました。\n"
             f"明日の取引で売却を検討してください。現在の指数 {value:+.2f}")
    print(f"{date} 指数 {value:+.3f} / シグナル {'あり' if crossed.iloc[-1] else 'なし'}")


if __name__ == "__main__":
    main()
