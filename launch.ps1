$Host.UI.RawUI.WindowTitle = "セクターローテーション ダッシュボード"
Set-Location $PSScriptRoot
Write-Host "============================================"
Write-Host " セクターローテーション ダッシュボード"
Write-Host "============================================"
Write-Host ""
Write-Host "[1/3] データ更新中…（4時間以内に取得済みならスキップ / オフラインなら前回データで表示）"
python fetch_data.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ※更新に失敗しました。前回取得したデータで表示します。"
}
Write-Host ""
Write-Host "[2/3] ダッシュボード生成中…"
python build_dashboard.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "  生成に失敗しました。一度もデータ取得ができていない可能性があります。"
    Write-Host "  ネットにつないだ状態でもう一度起動してください。"
    Read-Host "Enterキーで閉じます"
    exit 1
}
Write-Host ""
Write-Host "[3/3] ブラウザで開きます…"
Start-Process "$PSScriptRoot\out\index.html"
Start-Sleep -Seconds 2
