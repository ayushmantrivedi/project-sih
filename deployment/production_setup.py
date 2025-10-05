"""
Production deployment setup and configuration
"""
import os
import subprocess
import yaml
from pathlib import Path

def create_production_config():
    """Create production configuration files"""
    
    # Production Docker Compose
    docker_compose_prod = """
version: '3.8'

services:
  healthbot:
    build: .
    ports:
      - "5000:5000"
    env_file:
      - .env.production
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models
      - ./database:/app/database
      - ./augmented_synthetic_health_dataset.csv:/app/augmented_synthetic_health_dataset.csv
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./deployment/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - healthbot
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: healthbot
      POSTGRES_USER: healthbot
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U healthbot"]
      interval: 30s
      timeout: 10s
      retries: 3

  monitoring:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
"""
    
    # Production Nginx Configuration
    nginx_config = """
events {
    worker_connections 1024;
}

http {
    upstream healthbot {
        server healthbot:5000;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=webhook:10m rate=5r/s;
    
    # SSL Configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    server {
        listen 80;
        server_name yourdomain.com www.yourdomain.com;
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name yourdomain.com www.yourdomain.com;
        
        # SSL certificates
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        
        # API endpoints
        location /chat {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://healthbot;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Webhook endpoints
        location /webhook/ {
            limit_req zone=webhook burst=10 nodelay;
            proxy_pass http://healthbot;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Health check
        location /health {
            proxy_pass http://healthbot;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Monitoring
        location /metrics {
            proxy_pass http://healthbot;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
"""
    
    # Production Environment
    env_production = """
# Production Environment Variables
DEBUG=False
SECRET_KEY=your-super-secret-production-key-change-this
HOST=0.0.0.0
PORT=5000
LOG_LEVEL=INFO
ENABLE_CACHE=True

# Database Configuration
DATABASE_URL=postgresql://healthbot:${POSTGRES_PASSWORD}@postgres:5432/healthbot
REDIS_URL=redis://redis:6379/0

# Twilio Configuration
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
TWILIO_SMS_NUMBER=+1234567890

# Webhook URLs
WHATSAPP_WEBHOOK_URL=https://yourdomain.com/webhook/whatsapp
SMS_WEBHOOK_URL=https://yourdomain.com/webhook/sms

# API Keys
UMLS_API_KEY=your_umls_api_key
INFERMEDICA_APP_ID=your_infermedica_app_id
INFERMEDICA_APP_KEY=your_infermedica_app_key

# PostgreSQL Password
POSTGRES_PASSWORD=your_secure_postgres_password

# Monitoring
PROMETHEUS_ENABLED=True
PROMETHEUS_PORT=9090
"""
    
    # Systemd Service
    systemd_service = """
[Unit]
Description=HealthBot AI Chatbot
After=network.target

[Service]
Type=simple
User=healthbot
WorkingDirectory=/opt/healthbot
ExecStart=/usr/local/bin/docker-compose up
ExecStop=/usr/local/bin/docker-compose down
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    # Create directories
    os.makedirs("deployment", exist_ok=True)
    os.makedirs("monitoring", exist_ok=True)
    os.makedirs("ssl", exist_ok=True)
    
    # Write files
    with open("docker-compose.prod.yml", "w") as f:
        f.write(docker_compose_prod)
    
    with open("deployment/nginx.conf", "w") as f:
        f.write(nginx_config)
    
    with open(".env.production", "w") as f:
        f.write(env_production)
    
    with open("deployment/healthbot.service", "w") as f:
        f.write(systemd_service)
    
    print("✅ Production configuration files created!")

def create_ssl_certificates():
    """Create SSL certificates for production"""
    print("🔐 Creating SSL certificates...")
    
    # Create self-signed certificate for testing
    # In production, use Let's Encrypt or a proper CA
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:4096", 
        "-keyout", "ssl/key.pem", "-out", "ssl/cert.pem",
        "-days", "365", "-nodes", "-subj", 
        "/C=US/ST=State/L=City/O=Organization/CN=yourdomain.com"
    ])
    
    print("✅ SSL certificates created!")

def create_monitoring_config():
    """Create monitoring configuration"""
    prometheus_config = """
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'healthbot'
    static_configs:
      - targets: ['healthbot:5000']
    metrics_path: '/metrics'
    scrape_interval: 30s
"""
    
    with open("monitoring/prometheus.yml", "w") as f:
        f.write(prometheus_config)
    
    print("✅ Monitoring configuration created!")

def create_deployment_script():
    """Create deployment script"""
    deploy_script = """#!/bin/bash

# HealthBot AI Chatbot Production Deployment Script

set -e

echo "🚀 Starting HealthBot AI Chatbot deployment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
mkdir -p logs models database ssl

# Set up environment
if [ ! -f .env.production ]; then
    echo "❌ .env.production file not found. Please create it first."
    exit 1
fi

# Create SSL certificates if they don't exist
if [ ! -f ssl/cert.pem ] || [ ! -f ssl/key.pem ]; then
    echo "🔐 Creating SSL certificates..."
    ./deployment/create_ssl.sh
fi

# Build and start services
echo "🏗️ Building and starting services..."
docker-compose -f docker-compose.prod.yml up -d --build

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check health
echo "🏥 Checking service health..."
if curl -f http://localhost/health > /dev/null 2>&1; then
    echo "✅ HealthBot is running successfully!"
    echo "🌐 Web API: https://yourdomain.com"
    echo "📱 WhatsApp: Configure webhook to https://yourdomain.com/webhook/whatsapp"
    echo "📱 SMS: Configure webhook to https://yourdomain.com/webhook/sms"
    echo "📊 Monitoring: http://localhost:9090"
else
    echo "❌ HealthBot is not responding. Check logs:"
    docker-compose -f docker-compose.prod.yml logs
    exit 1
fi

echo "🎉 Deployment completed successfully!"
"""
    
    ssl_script = """#!/bin/bash

# SSL Certificate Creation Script

echo "🔐 Creating SSL certificates..."

# Create self-signed certificate
openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=yourdomain.com"

echo "✅ SSL certificates created!"
echo "⚠️  Note: These are self-signed certificates. For production, use Let's Encrypt or a proper CA."
"""
    
    with open("deploy.sh", "w") as f:
        f.write(deploy_script)
    
    with open("deployment/create_ssl.sh", "w") as f:
        f.write(ssl_script)
    
    # Make scripts executable
    os.chmod("deploy.sh", 0o755)
    os.chmod("deployment/create_ssl.sh", 0o755)
    
    print("✅ Deployment scripts created!")

def main():
    """Main setup function"""
    print("🏗️ Setting up production deployment...")
    
    create_production_config()
    create_ssl_certificates()
    create_monitoring_config()
    create_deployment_script()
    
    print("\n🎉 Production setup completed!")
    print("\n📋 Next steps:")
    print("1. Update .env.production with your actual credentials")
    print("2. Update nginx.conf with your domain name")
    print("3. Run ./deploy.sh to deploy")
    print("4. Configure Twilio webhooks to point to your domain")
    print("5. Set up SSL certificates (Let's Encrypt recommended)")

if __name__ == "__main__":
    main()