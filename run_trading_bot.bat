@echo off
setlocal enabledelayedexpansion

echo Starting Algo Trading Bot...

:: Default to NIFTY 50 (1) if no argument provided
set INSTRUMENT=1
if not "%1"=="" set INSTRUMENT=%1

:: Get the current directory
set "CURRENT_DIR=%~dp0"
set "CURRENT_DIR=%CURRENT_DIR:~0,-1%"

:: Format date for log file (YYYYMMDD)
for /f %%I in ('%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe -Command "Get-Date -Format 'yyyyMMdd'"') do set LOG_DATE=%%I

:: Check if Docker Desktop is running
tasklist /FI "IMAGENAME eq Docker Desktop.exe" 2>NUL | find /I /N "Docker Desktop.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo Docker Desktop is not running. Starting Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    timeout /t 30 /nobreak
)

:: Check and setup WSL Docker integration
echo Checking WSL Docker integration...
wsl -d Ubuntu sh -c "if ! command -v docker >/dev/null 2>&1; then echo 'Docker is installed in WSL'; else echo 'Setting up Docker in WSL...' && sudo apt-get update && sudo apt-get install -y docker.io && sudo usermod -aG docker $USER; fi"

:: Check if Docker is running in WSL
wsl -d Ubuntu sh -c "if ! systemctl is-active --quiet docker; then echo 'Starting Docker...' && sudo service docker start; fi"

:: Run the trading bot
echo Running trading bot for instrument: %INSTRUMENT%
wsl -d Ubuntu sh -c "cd /mnt/c/Users/rajiv/OneDrive/PERSONAL/Desktop/MS && docker compose down && docker compose up -d --build && timeout 5 && docker compose exec -T trading_bot python AlgoTrade.py %INSTRUMENT% > logs/trading_%LOG_DATE%.log 2>&1"

:: Check if the command was successful
if %ERRORLEVEL% NEQ 0 (
    echo Error: Trading bot failed to start
    echo Please check the logs for more information
    pause
    exit /b 1
)

echo Trading bot started successfully
echo Log file: logs\trading_%LOG_DATE%.log

:: Start Flask dashboard with Gunicorn
echo Starting Flask dashboard with Gunicorn...
start cmd /k "cd %CURRENT_DIR% && .venv\Scripts\gunicorn -w 4 -b 0.0.0.0:5000 app:app"

echo Dashboard started at http://localhost:5000
echo Press any key to exit...
pause 