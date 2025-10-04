# 🐳 Docker Setup Guide for HealthBot AI Chatbot

This guide will help you run the HealthBot AI Chatbot using Docker on Windows.

## 📋 Prerequisites

1. **Docker Desktop for Windows**
   - Download from: https://www.docker.com/products/docker-desktop
   - Install and restart your computer
   - Start Docker Desktop

2. **Required Files**
   - All project files should be in your `sihdemo` directory
   - Dataset file: `augmented_synthetic_health_dataset.csv`

## 🚀 Quick Start (Automated)

### Option 1: PowerShell Script (Recommended)
```powershell
# Run the automated setup script
.\setup_docker.ps1
```

### Option 2: Batch File
```cmd
# Run the automated setup script
setup_docker.bat
```

## 🔧 Manual Setup

If you prefer to run the commands manually:

### 1. Check Docker Installation
```cmd
docker --version
docker-compose --version
```

### 2. Build and Start Services
```cmd
# Build the Docker image
docker-compose build --no-cache

# Start services in background
docker-compose up -d
```

### 3. Check Service Status
```cmd
# View running containers
docker-compose ps

# Check health
curl http://localhost:5000/health
```

## 🌐 Access Your Application

Once running, your HealthBot will be available at:

- **Web API**: http://localhost:5000
- **Health Check**: http://localhost:5000/health
- **Chat API**: POST http://localhost:5000/chat
- **WhatsApp Webhook**: POST http://localhost:5000/webhook/whatsapp
- **SMS Webhook**: POST http://localhost:5000/webhook/sms

## 📱 Test the Chatbot

### Test with curl:
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I have fever and cough", "user_id": "test_user"}'
```

### Test with PowerShell:
```powershell
$body = @{
    message = "I have fever and cough"
    user_id = "test_user"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/chat" -Method POST -Body $body -ContentType "application/json"
```

## 🔧 Configuration

### Edit Environment Variables
1. Open `docker.env` file
2. Update the following values:
   ```env
   # Twilio Configuration (Required for WhatsApp/SMS)
   TWILIO_ACCOUNT_SID=your_actual_account_sid
   TWILIO_AUTH_TOKEN=your_actual_auth_token
   
   # API Keys (Optional)
   UMLS_API_KEY=your_umls_api_key
   INFERMEDICA_APP_ID=your_infermedica_app_id
   INFERMEDICA_APP_KEY=your_infermedica_app_key
   ```

3. Restart services after changes:
   ```cmd
   docker-compose restart
   ```

## 📋 Useful Commands

### View Logs
```cmd
# View all logs
docker-compose logs -f

# View only healthbot logs
docker-compose logs -f healthbot

# View only redis logs
docker-compose logs -f redis
```

### Manage Services
```cmd
# Stop all services
docker-compose down

# Stop and remove volumes (clears data)
docker-compose down -v

# Restart services
docker-compose restart

# Rebuild and restart
docker-compose up --build -d
```

### Service Status
```cmd
# Check running containers
docker-compose ps

# Check resource usage
docker stats

# Check container health
docker-compose exec healthbot curl http://localhost:5000/health
```

## 🔍 Troubleshooting

### Common Issues

1. **Port 5000 already in use**
   ```cmd
   # Change port in docker-compose.yml
   ports:
     - "5001:5000"  # Use port 5001 instead
   ```

2. **Docker build fails**
   ```cmd
   # Clean build
   docker-compose down
   docker system prune -f
   docker-compose build --no-cache
   ```

3. **Services not starting**
   ```cmd
   # Check logs
   docker-compose logs healthbot
   
   # Check if Docker Desktop is running
   docker info
   ```

4. **Health check fails**
   ```cmd
   # Wait longer for services to start
   docker-compose up -d
   # Wait 30 seconds
   curl http://localhost:5000/health
   ```

### Reset Everything
```cmd
# Stop and remove everything
docker-compose down -v
docker system prune -a -f

# Rebuild from scratch
docker-compose build --no-cache
docker-compose up -d
```

## 📊 Monitoring

### Health Monitoring
- Health endpoint: http://localhost:5000/health
- Docker health checks are configured
- Logs are stored in `./logs/` directory

### Performance
- Application runs with 4 workers by default
- Redis is used for caching
- Logs are rotated automatically

## 🚀 Production Deployment

For production deployment:

1. **Update environment variables** in `docker.env`
2. **Set up SSL certificates** and reverse proxy (nginx)
3. **Configure domain names** and webhook URLs
4. **Set up monitoring** and alerting
5. **Configure backup** for database and logs

## 📞 Support

If you encounter issues:

1. Check the logs: `docker-compose logs -f healthbot`
2. Verify Docker Desktop is running
3. Ensure all required files are present
4. Check if ports 5000 and 6379 are available

## 🎉 Success!

Once everything is running, you'll see:
```
🎉 HealthBot AI Chatbot is now running!
🌐 Web API: http://localhost:5000
❤️  Health Check: http://localhost:5000/health
```

Your AI-powered health chatbot is ready to help users with symptom analysis, medical information, and health data!

