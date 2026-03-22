$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$targets = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -eq "python.exe" -and
        $_.CommandLine -like "*paperbanana2.0*" -and
        $_.CommandLine -like "*streamlit run demo.py*"
    }

if (-not $targets) {
    Write-Output "PaperBanana is not running."
    exit 0
}

$targets | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 2

$remaining = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -eq "python.exe" -and
        $_.CommandLine -like "*paperbanana2.0*" -and
        $_.CommandLine -like "*streamlit run demo.py*"
    }

if ($remaining) {
    Write-Output "Some PaperBanana processes are still present:"
    $remaining | Select-Object ProcessId, ParentProcessId, Name, CommandLine
    exit 1
}

Write-Output "PaperBanana stopped successfully."
