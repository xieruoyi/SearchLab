$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:HF_HOME = Join-Path $projectRoot '.cache\huggingface'
$env:TEMP = Join-Path $projectRoot '.cache\tmp'
$env:TMP = $env:TEMP
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force reports, experiments, $env:TEMP | Out-Null
& $projectPython -m pip freeze | Set-Content -Encoding utf8 requirements.lock.txt
& $projectPython scripts\test_metrics.py
if ($LASTEXITCODE -ne 0) { throw 'Metric validation failed' }
foreach ($stage in @('retrieve', 'smoke', 'rerank', 'report')) {
    & $projectPython -u scripts\experiment.py $stage
    if ($LASTEXITCODE -ne 0) { throw "Experiment stage failed: $stage" }
}
& $projectPython scripts\validate_results.py
if ($LASTEXITCODE -ne 0) { throw 'Result integrity validation failed' }
