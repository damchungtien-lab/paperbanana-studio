$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment not found at $python"
}

$port = 8501
$url = "http://127.0.0.1:$port"
$logDir = Join-Path $projectRoot "logs"
$stdout = Join-Path $logDir "streamlit_stdout.log"
$stderr = Join-Path $logDir "streamlit_stderr.log"

function Get-PaperBananaProcesses {
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -eq "python.exe" -and
            $_.CommandLine -like "*paperbanana2.0*" -and
            $_.CommandLine -like "*streamlit run demo.py*"
        }
}

function Test-PaperBananaHttp {
    try {
        $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
        return $resp.StatusCode -eq 200
    } catch {
        return $false
    }
}

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$existing = Get-PaperBananaProcesses
if ($existing) {
    if (Test-PaperBananaHttp) {
        Write-Output "PaperBanana is already running at $url"
        $existing | Select-Object ProcessId, ParentProcessId, Name, CommandLine
        exit 0
    }

    Write-Output "Found stale PaperBanana processes. Cleaning up before restart..."
    $existing | ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}

if (Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue) {
    Write-Output "Port $port is occupied by another process. Showing current listeners:"
    Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
        Select-Object LocalAddress, LocalPort, State, OwningProcess
    exit 1
}

if (Test-Path $stdout) { Remove-Item $stdout -Force }
if (Test-Path $stderr) { Remove-Item $stderr -Force }

$proc = Start-Process `
    -FilePath $python `
    -ArgumentList @("-m", "streamlit", "run", "demo.py", "--server.headless", "true", "--server.port", "$port") `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

for ($i = 0; $i -lt 15; $i++) {
    Start-Sleep -Seconds 2

    if ($proc.HasExited) {
        Write-Output "PaperBanana process exited unexpectedly with code $($proc.ExitCode)."
        if (Test-Path $stderr) {
            Write-Output "stderr:"
            Get-Content $stderr
        }
        exit 1
    }

    if (Test-PaperBananaHttp) {
        Write-Output "PaperBanana started successfully at $url"
        Write-Output "PID: $($proc.Id)"
        exit 0
    }
}

Write-Output "PaperBanana process is running, but the HTTP health check did not pass in time."
Write-Output "stdout log: $stdout"
Write-Output "stderr log: $stderr"
exit 1
