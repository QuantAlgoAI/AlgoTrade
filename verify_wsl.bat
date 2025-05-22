@echo off
echo Checking WSL installation and configuration...

:: Check WSL version
echo.
echo Checking WSL version...
wsl --version

:: List installed distributions
echo.
echo Listing installed WSL distributions...
wsl --list --verbose

:: Check Docker in WSL
echo.
echo Checking Docker in WSL...
wsl -d Ubuntu sh -c "docker --version && docker-compose --version"

:: Check Python in WSL
echo.
echo Checking Python in WSL...
wsl -d Ubuntu sh -c "python3 --version && pip3 --version"

:: Check project directory
echo.
echo Checking project directory...
wsl -d Ubuntu sh -c "ls -la /mnt/c/Users/rajiv/OneDrive/PERSONAL/Desktop/MS"

echo.
echo Verification complete. Press any key to exit...
pause 