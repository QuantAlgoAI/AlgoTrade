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
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set "LOG_DATE=%datetime:~0,8%"

:: Check if Docker is running in WSL
wsl -d Ubuntu sh -c "if ! systemctl is-active --quiet docker; then echo 'Starting Docker...' && sudo service docker start; fi"

:: Run the trading bot
echo Running trading bot for instrument: %INSTRUMENT%
wsl -d Ubuntu sh -c "cd /mnt/c/Users/rajiv/OneDrive/PERSONAL/Desktop/MS && \
    docker-compose down && \
    docker-compose up -d --build && \
    sleep 5 && \
    docker-compose exec -T trading_bot python AlgoTrade.py %INSTRUMENT% 2>&1 | tee logs/trading_%LOG_DATE%.log"

:: Check if the command was successful
if %ERRORLEVEL% NEQ 0 (
    echo Error: Trading bot failed to start
    echo Please check the logs for more information
    pause
    exit /b 1
)

echo Trading bot started successfully
echo Log file: logs\trading_%LOG_DATE%.log

:: Keep the window open
pause 