# 日経225 セクターローテーション ダッシュボード

日経225の全採用銘柄（225銘柄）を21のテーマ別バスケットに分け、セクター間の資金循環を可視化する。
毎営業日の夜に GitHub Actions が自動でデータ取得→ページ更新し、GitHub Pages で公開される
（StatiCrypt によるパスワード保護付き）。

## 構成

| ファイル | 役割 |
|---|---|
| `baskets.json` | 基本層＝セクター定義（重複なし・日経225のみ）。市場平均とローテーション指数の計算根拠 |
| `themes.json` | テーマ層＝投資テーマ定義（重複可・225外の銘柄も含む）。表示専用で指数計算には使わない |
| `fetch_members.py` | 日経公式サイトから採用225銘柄と業種を取得 → `data/nikkei225_members.json` |
| `make_baskets.py` | 採用銘柄リストを21バスケットに振り分け → `baskets.json`（テーマの上書きは OVERRIDE を編集） |
| `fetch_data.py` | J-Quants API V2 から全銘柄の日足を取得 → `data/prices.parquet` |
| `build_dashboard.py` | 指標計算 + Plotly ダッシュボード生成 → `out/index.html` |
| `.github/workflows/update.yml` | 毎営業日 20:30 JST に自動実行（取得→生成→暗号化→公開） |

日経平均の銘柄入れ替え（4月・10月の定期見直しなど）の後は
`python fetch_members.py && python make_baskets.py` を実行して push すれば反映される。

## 2層構造

| 層 | 対象 | 重複 | 用途 |
|---|---|---|---|
| 基本層（24セクター） | 日経225の225銘柄 | なし（完全分割） | 市場平均・ローテーション指数の計算＋表示 |
| テーマ層（27テーマ） | 日経225＋225外の約88銘柄 | あり | 表示のみ（指数計算には不使用） |

テーマ層の市場平均は基本層のものを流用する（テーマ同士は銘柄が重複するため独自の平均が取れないため）。
テーマの追加・銘柄の入れ替えは `themes.json` の編集だけで反映され、取得対象銘柄も自動で追随する。

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

## ローカルで動かす場合（PCアプリ版）

**`起動.bat` をダブルクリック**（デスクトップのショートカット「セクターローテーション」からも起動可）。
データ更新 → ダッシュボード生成 → ブラウザ表示まで自動で行う。

- 4時間以内に取得済みならデータ更新はスキップ（強制更新は `python fetch_data.py --force`）
- オフラインのときは前回取得したデータで表示される
- ローカル版はパスワード無し。公開Web版とは独立して動く

手動で個別に実行する場合:

```
# .env に JQUANTS_API_KEY=... を書いておく
pip install -r requirements.txt
python fetch_data.py
python build_dashboard.py
# out/index.html をブラウザで開く
```

## 注意

- J-Quants の生データ（株価そのもの）はリポジトリにコミットしない・ページに載せない
  （利用規約の再配布禁止に配慮。公開するのは加工済みの独自指標のみ）
- 日経225の銘柄入れ替えは `fetch_members.py` → `make_baskets.py`、テーマの組み替えは `make_baskets.py` の OVERRIDE 編集で対応
