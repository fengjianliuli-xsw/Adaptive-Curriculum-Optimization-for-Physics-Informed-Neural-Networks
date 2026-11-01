$dashboardPath = Join-Path $PSScriptRoot 'runs\baseline\dashboard.html'
if (-not (Test-Path -LiteralPath $dashboardPath)) {
    throw 'Run train.py first; see README.md.'
}
Start-Process -FilePath $dashboardPath
