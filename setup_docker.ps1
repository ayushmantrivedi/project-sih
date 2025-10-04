# HealthBot AI Chatbot Docker Setup Script for Windows
# This script helps you install Docker and run the HealthBot application

Write-Host "🚀 HealthBot AI Chatbot Docker Setup" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green

# Check if Docker is installed
Write-Host "`n🔍 Checking Docker installation..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version 2>$null
    if ($dockerVersion) {
        Write-Host "✅ Docker is already installed: $dockerVersion" -ForegroundColor Green
        $dockerInstalled = $true
    }
} catch {
    $dockerInstalled = $false
}

if (-not $dockerInstalled) {
    Write-Host "❌ Docker is not installed." -ForegroundColor Red
    Write-Host "`n📥 Please install Docker Desktop for Windows:" -ForegroundColor Yellow
    Write-Host "1. Visit: https://www.docker.com/products/docker-desktop" -ForegroundColor Cyan
    Write-Host "2. Download Docker Desktop for Windows" -ForegroundColor Cyan
    Write-Host "3. Run the installer and restart your computer" -ForegroundColor Cyan
    Write-Host "4. Start Docker Desktop" -ForegroundColor Cyan
    Write-Host "`n⏳ After installing Docker, run this script again." -ForegroundColor Yellow
    
    # Open Docker download page
    $response = Read-Host "`nWould you like to open the Docker download page now? (y/n)"
    if ($response -eq 'y' -or $response -eq 'Y') {
        Start-Process "https://www.docker.com/products/docker-desktop"
    }
    exit 1
}

# Check if Docker is running
Write-Host "`n🔍 Checking if Docker is running..." -ForegroundColor Yellow
try {
    $dockerInfo = docker info 2>$null
    if ($dockerInfo) {
        Write-Host "✅ Docker is running" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Check if docker-compose is available
Write-Host "`n🔍 Checking Docker Compose..." -ForegroundColor Yellow
try {
    $composeVersion = docker-compose --version 2>$null
    if ($composeVersion) {
        Write-Host "✅ Docker Compose is available: $composeVersion" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Docker Compose not found. Please install Docker Desktop with Docker Compose." -ForegroundColor Red
    exit 1
}

# Check if required files exist
Write-Host "`n🔍 Checking required files..." -ForegroundColor Yellow
$requiredFiles = @(
    "Dockerfile",
    "docker-compose.yml",
    "docker.env",
    "requirements.txt"
)

$missingFiles = @()
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "✅ Found: $file" -ForegroundColor Green
    } else {
        Write-Host "❌ Missing: $file" -ForegroundColor Red
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host "`n❌ Missing required files. Please ensure all files are present." -ForegroundColor Red
    exit 1
}

# Check if dataset exists
Write-Host "`n🔍 Checking dataset file..." -ForegroundColor Yellow
if (Test-Path "augmented_synthetic_health_dataset.csv") {
    Write-Host "✅ Dataset found: augmented_synthetic_health_dataset.csv" -ForegroundColor Green
} else {
    Write-Host "⚠️  Warning: Dataset file not found. The application will work but ML features may not be available." -ForegroundColor Yellow
}

# Build and run the application
Write-Host "`n🏗️  Building and starting HealthBot..." -ForegroundColor Yellow

try {
    # Build the Docker image
    Write-Host "Building Docker image..." -ForegroundColor Cyan
    docker-compose build --no-cache
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Docker image built successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to build Docker image" -ForegroundColor Red
        exit 1
    }
    
    # Start the services
    Write-Host "Starting services..." -ForegroundColor Cyan
    docker-compose up -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Services started successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to start services" -ForegroundColor Red
        exit 1
    }
    
} catch {
    Write-Host "❌ Error during build/start: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Wait a moment for services to start
Write-Host "`n⏳ Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check service status
Write-Host "`n🔍 Checking service status..." -ForegroundColor Yellow
docker-compose ps

# Test health endpoint
Write-Host "`n🏥 Testing health endpoint..." -ForegroundColor Yellow
try {
    $healthResponse = Invoke-RestMethod -Uri "http://localhost:5000/health" -Method GET -TimeoutSec 10
    Write-Host "✅ Health check passed: $($healthResponse.status)" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Health check failed or service not ready yet" -ForegroundColor Yellow
    Write-Host "You can check the logs with: docker-compose logs healthbot" -ForegroundColor Cyan
}

# Display success message
Write-Host "`n🎉 HealthBot AI Chatbot is now running!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host "🌐 Web API: http://localhost:5000" -ForegroundColor Cyan
Write-Host "❤️  Health Check: http://localhost:5000/health" -ForegroundColor Cyan
Write-Host "💬 Chat API: POST http://localhost:5000/chat" -ForegroundColor Cyan
Write-Host "📱 WhatsApp Webhook: POST http://localhost:5000/webhook/whatsapp" -ForegroundColor Cyan
Write-Host "📧 SMS Webhook: POST http://localhost:5000/webhook/sms" -ForegroundColor Cyan

Write-Host "`n📋 Useful Commands:" -ForegroundColor Yellow
Write-Host "  View logs: docker-compose logs -f healthbot" -ForegroundColor White
Write-Host "  Stop services: docker-compose down" -ForegroundColor White
Write-Host "  Restart services: docker-compose restart" -ForegroundColor White
Write-Host "  View service status: docker-compose ps" -ForegroundColor White

Write-Host "`n🔧 Configuration:" -ForegroundColor Yellow
Write-Host "  Edit docker.env to configure API keys and settings" -ForegroundColor White
Write-Host "  After editing, run: docker-compose restart" -ForegroundColor White

Write-Host "`n✨ Your HealthBot AI Chatbot is ready to use!" -ForegroundColor Green

