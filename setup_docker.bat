@echo off
echo 🚀 HealthBot AI Chatbot Docker Setup
echo =====================================

echo.
echo 🔍 Checking Docker installation...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed.
    echo.
    echo 📥 Please install Docker Desktop for Windows:
    echo 1. Visit: https://www.docker.com/products/docker-desktop
    echo 2. Download Docker Desktop for Windows
    echo 3. Run the installer and restart your computer
    echo 4. Start Docker Desktop
    echo.
    echo ⏳ After installing Docker, run this script again.
    pause
    exit /b 1
)

echo ✅ Docker is installed

echo.
echo 🔍 Checking if Docker is running...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)

echo ✅ Docker is running

echo.
echo 🔍 Checking required files...
if not exist "Dockerfile" (
    echo ❌ Missing: Dockerfile
    goto :error
)
if not exist "docker-compose.yml" (
    echo ❌ Missing: docker-compose.yml
    goto :error
)
if not exist "docker.env" (
    echo ❌ Missing: docker.env
    goto :error
)
if not exist "requirements.txt" (
    echo ❌ Missing: requirements.txt
    goto :error
)

echo ✅ All required files found

echo.
echo 🏗️  Building and starting HealthBot...
echo Building Docker image...
docker-compose build --no-cache
if %errorlevel% neq 0 (
    echo ❌ Failed to build Docker image
    pause
    exit /b 1
)

echo ✅ Docker image built successfully

echo Starting services...
docker-compose up -d
if %errorlevel% neq 0 (
    echo ❌ Failed to start services
    pause
    exit /b 1
)

echo ✅ Services started successfully

echo.
echo ⏳ Waiting for services to start...
timeout /t 10 /nobreak >nul

echo.
echo 🔍 Checking service status...
docker-compose ps

echo.
echo 🎉 HealthBot AI Chatbot is now running!
echo =====================================
echo 🌐 Web API: http://localhost:5000
echo ❤️  Health Check: http://localhost:5000/health
echo 💬 Chat API: POST http://localhost:5000/chat
echo 📱 WhatsApp Webhook: POST http://localhost:5000/webhook/whatsapp
echo 📧 SMS Webhook: POST http://localhost:5000/webhook/sms

echo.
echo 📋 Useful Commands:
echo   View logs: docker-compose logs -f healthbot
echo   Stop services: docker-compose down
echo   Restart services: docker-compose restart
echo   View service status: docker-compose ps

echo.
echo ✨ Your HealthBot AI Chatbot is ready to use!
pause
exit /b 0

:error
echo ❌ Missing required files. Please ensure all files are present.
pause
exit /b 1

