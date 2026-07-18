# 日経225 セクターローテーション ダッシュボード

日経225の全採用銘柄（225銘柄）を21のテーマ別バスケットに分け、セクター間の資金循環を可視化する。
毎営業日の夜に GitHub Actions が自動でデータ取得→ページ更新し、GitHub Pages で公開される
（StatiCrypt によるパスワード保護付き）。

## 構成

| ファイル | 役割 |
|---|---|
| `baskets.json` | バスケット定義（手編集も可。通常は下の2スクリプトで再生成） |
| `fetch_members.py` | 日経公式サイトから採用225銘柄と業種を取得 → `data/nikkei225_members.json` |
| `make_baskets.py` | 採用銘柄リストを21バスケットに振り分け → `baskets.json`（テーマの上書きは OVERRIDE を編集） |
| `fetch_data.py` | J-Quants API V2 から全銘柄の日足を取得 → `data/prices.parquet` |
| `build_dashboard.py` | 指標計算 + Plotly ダッシュボード生成 → `out/index.html` |
| `.github/workflows/update.yml` | 毎営業日 20:30 JST に自動実行（取得→生成→暗号化→公開） |

日経平均の銘柄入れ替え（4月・10月の定期見直しなど）の後は
`python fetch_members.py && python make_baskets.py` を実行して push すれば反映される。

## 指標の定義

- **バスケット日次リターン** = 構成銘柄の対数リターンの単純平均
- **相対リターン** = バスケットリターン − 全バスケット平均（市場）
- **累積相対強弱 RS** = 相対リターンの累積和（市場に対する勝ち負け）
- **資金フロー** = 相対リターンをガウス平滑化したもの（%/日）。プラス＝流入
- **ローテーション指数** = 攻めバスケットのフロー平均 − 守り（ディフェンシブ・通信）のフロー平均
  - ゼロクロス＝攻守転換、±σブレイク＝行き過ぎ、加速/減速＝2階差分のしきい値超え

## 初回セットアップ（1回だけ）

1. GitHub に **公開リポジトリ** `sector-rotation` を作って、このフォルダを push する
2. リポジトリの **Settings → Secrets and variables → Actions → New repository secret** で2つ登録：
   - `JQUANTS_API_KEY` … J-Quants の API キー
   - `PAGE_PASSWORD` … ページを開くときのパスワード（自分で決めた文字列）
3. **Settings → Pages** で Source を **GitHub Actions** に変更する
4. **Actions タブ → update-dashboard → Run workflow** で初回実行
5. 完了すると `https://<ユーザー名>.github.io/sector-rotation/` で閲覧できる

## ローカルで動かす場合

```
# .env に JQUANTS_API_KEY=... を書いておく
pip install -r requirements.txt
python fetch_data.py
python build_dashboard.py
# out/index.html をブラウザで開く（ローカル版はパスワード無し）
```

## 注意

- J-Quants の生データ（株価そのもの）はリポジトリにコミットしない・ページに載せない
  （利用規約の再配布禁止に配慮。公開するのは加工済みの独自指標のみ）
- 日経225の銘柄入れ替えは `fetch_members.py` → `make_baskets.py`、テーマの組み替えは `make_baskets.py` の OVERRIDE 編集で対応
