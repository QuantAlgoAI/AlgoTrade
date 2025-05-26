@echo off
echo Checking WSL installation and configuration...

:: Check WSL version
echo.
echo Checking WSL version...
wsl --version

:: List WSL distributions
echo.
echo Listing installed WSL distributions...
wsl --list --verbose

:: Check Docker in WSL
echo.
echo Checking Docker in WSL...
wsl -d Ubuntu sh -c "docker --version || echo 'Docker not found in WSL'"
wsl -d Ubuntu sh -c "docker compose version || echo 'Docker Compose not found in WSL'"

:: Check Docker service
echo.
echo Checking Docker service status...
wsl -d Ubuntu sh -c "sudo systemctl status docker | grep Active || echo 'Docker service not running'"

:: Check Python in WSL
echo.
echo Checking Python in WSL...
wsl -d Ubuntu sh -c "python3 --version && pip3 --version || echo 'Python/pip not found in WSL'"

:: Check project directory
echo.
echo Checking project directory...
wsl -d Ubuntu sh -c "cd /mnt/c/Users/rajiv/OneDrive/PERSONAL/Desktop/MS && ls -la"

:: Check Docker group membership
echo.
echo Checking Docker group membership...
wsl -d Ubuntu sh -c "groups | grep docker || echo 'User not in docker group'"

echo.
echo Verification complete. Press any key to exit...
pause 