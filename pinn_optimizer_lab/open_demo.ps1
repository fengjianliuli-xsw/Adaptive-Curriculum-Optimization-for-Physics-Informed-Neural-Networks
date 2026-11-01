$dashboardPath = Join-Path $PSScriptRoot 'runs\demo\dashboard.html'
if (-not (Test-Path -LiteralPath $dashboardPath)) {
    throw 'Demo is missing. Run train.py first; see README.md.'
}
Start-Process -FilePath $dashboardPath
