@echo off
echo Setting up WSL environment...

:: Check if WSL is installed
wsl --status >nul 2>&1
if %errorlevel% neq 0 (
    echo WSL is not installed. Installing WSL...
    wsl --install
    echo Please restart your computer and run this script again.
    pause
    exit /b
)

:: List available distributions
echo.
echo Available WSL distributions:
wsl --list --online

:: Install Ubuntu if not present
echo.
echo Checking for Ubuntu distribution...
wsl --list | findstr "Ubuntu" >nul
if %errorlevel% neq 0 (
    echo Installing Ubuntu...
    wsl --install -d Ubuntu
    echo Please set up your Ubuntu username and password when prompted.
    pause
    exit /b
)

:: Update WSL
echo.
echo Updating WSL...
wsl --update

:: Install required packages in Ubuntu
echo.
echo Installing required packages in Ubuntu...
wsl -d Ubuntu sh -c "sudo apt-get update && \
    sudo apt-get install -y docker.io docker-compose python3 python3-pip && \
    sudo usermod -aG docker $USER"

echo.
echo Setup complete. Please restart your computer and run verify_wsl.bat to check the installation.
pause 