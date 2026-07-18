$ErrorActionPreference = "Stop"

$uv = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
if ($null -eq $uv) {
    Write-Error "'uv' is required; install it and ensure it is on PATH"
    exit 1
}

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$manager = Join-Path $PSScriptRoot "web_ui_process.py"
$exitCode = 1
Push-Location $repositoryRoot
try {
    & $uv.Source run --no-sync python $manager start @args
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $exitCode
