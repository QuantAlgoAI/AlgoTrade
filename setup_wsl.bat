@echo off
echo Setting up WSL environment...

:: Start Ubuntu WSL distro
echo Starting Ubuntu WSL...
wsl -d Ubuntu

:: Install Docker in Ubuntu
echo Installing Docker in Ubuntu...
wsl -d Ubuntu sh -c "sudo apt-get update && \
    sudo apt-get install -y ca-certificates curl gnupg && \
    sudo install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    sudo chmod a+r /etc/apt/keyrings/docker.gpg && \
    echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu jammy stable' | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    sudo apt-get update && \
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin && \
    sudo usermod -aG docker $USER && \
    sudo systemctl enable docker && \
    sudo systemctl start docker"

:: Install pip in Ubuntu
echo Installing pip in Ubuntu...
wsl -d Ubuntu sh -c "sudo apt-get update && sudo apt-get install -y python3-pip"

:: Enable Docker service
echo Enabling Docker service...
wsl -d Ubuntu sh -c "sudo systemctl enable docker && sudo systemctl start docker"

:: Verify installation
echo Verifying installation...
wsl -d Ubuntu sh -c "docker --version && docker compose version && pip3 --version"

echo Setup complete. Please restart your terminal and run verify_wsl.bat to confirm the setup.
pause 