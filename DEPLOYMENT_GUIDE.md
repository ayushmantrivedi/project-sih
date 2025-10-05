# 🚀 HealthBot AI Chatbot - Full-Scale Deployment Guide

This guide will help you deploy your HealthBot AI Chatbot to production with WhatsApp and SMS integration.

## 📋 Prerequisites

### 1. **Twilio Account Setup**
- Create account at [Twilio Console](https://console.twilio.com/)
- Get your Account SID and Auth Token
- Set up WhatsApp Sandbox (for testing) or WhatsApp Business API (for production)
- Purchase a phone number for SMS

### 2. **Server Requirements**
- **Minimum**: 2 CPU cores, 4GB RAM, 20GB storage
- **Recommended**: 4 CPU cores, 8GB RAM, 50GB storage
- Ubuntu 20.04+ or CentOS 8+
- Docker and Docker Compose installed

### 3. **Domain and SSL**
- Domain name pointing to your server
- SSL certificate (Let's Encrypt recommended)

## 🛠️ Step-by-Step Deployment

### **Step 1: Server Setup**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Logout and login to apply Docker group changes
```

### **Step 2: Clone and Setup Project**

```bash
# Clone your repository
git clone <your-repo-url>
cd healthbot-ai-chatbot

# Create production environment
cp .env.example .env.production

# Edit production environment
nano .env.production
```

**Update `.env.production` with your credentials:**
```env
# Application Settings
DEBUG=False
SECRET_KEY=your-super-secret-production-key
HOST=0.0.0.0
PORT=5000

# Database Configuration
DATABASE_URL=postgresql://healthbot:your_password@postgres:5432/healthbot
REDIS_URL=redis://redis:6379/0

# Twilio Configuration
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
TWILIO_SMS_NUMBER=+1234567890

# Webhook URLs (Update with your domain)
WHATSAPP_WEBHOOK_URL=https://yourdomain.com/webhook/whatsapp
SMS_WEBHOOK_URL=https://yourdomain.com/webhook/sms

# API Keys (Optional)
UMLS_API_KEY=your_umls_api_key
INFERMEDICA_APP_ID=your_infermedica_app_id
INFERMEDICA_APP_KEY=your_infermedica_app_key

# PostgreSQL Password
POSTGRES_PASSWORD=your_secure_postgres_password
```

### **Step 3: SSL Certificate Setup**

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot certonly --standalone -d yourdomain.com -d www.yourdomain.com

# Copy certificates to project
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ssl/cert.pem
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ssl/key.pem
sudo chown $USER:$USER ssl/*.pem
```

### **Step 4: Deploy Application**

```bash
# Make deployment script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

### **Step 5: Configure Twilio Webhooks**

1. **WhatsApp Configuration:**
   - Go to [Twilio Console > Messaging > WhatsApp](https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn)
   - Set webhook URL: `https://yourdomain.com/webhook/whatsapp`
   - Set HTTP method: POST

2. **SMS Configuration:**
   - Go to [Twilio Console > Phone Numbers](https://console.twilio.com/us1/develop/phone-numbers/manage/incoming)
   - Select your phone number
   - Set webhook URL: `https://yourdomain.com/webhook/sms`
   - Set HTTP method: POST

### **Step 6: Test Integration**

```bash
# Test health endpoint
curl https://yourdomain.com/health

# Test chat API
curl -X POST https://yourdomain.com/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I have fever and cough", "user_id": "test_user"}'

# Test WhatsApp (send message to your WhatsApp number)
# Test SMS (send SMS to your Twilio number)
```

## 🔧 Configuration Files

### **Nginx Configuration** (`deployment/nginx.conf`)
```nginx
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
        }
    }
}
```

### **Docker Compose Production** (`docker-compose.prod.yml`)
```yaml
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
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

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
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: healthbot
      POSTGRES_USER: healthbot
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
```

## 📊 Monitoring and Maintenance

### **Health Monitoring**
```bash
# Check service status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f healthbot

# Check health endpoint
curl https://yourdomain.com/health
```

### **Performance Monitoring**
- **Prometheus**: http://yourdomain.com:9090
- **Grafana**: Set up for advanced monitoring
- **Logs**: Check `./logs/` directory

### **Backup Strategy**
```bash
# Backup database
docker-compose -f docker-compose.prod.yml exec postgres pg_dump -U healthbot healthbot > backup_$(date +%Y%m%d).sql

# Backup logs
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/

# Backup models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/
```

## 🔒 Security Best Practices

### **1. Environment Security**
- Use strong, unique passwords
- Rotate API keys regularly
- Enable 2FA on all accounts
- Use environment variables for secrets

### **2. Network Security**
- Configure firewall rules
- Use HTTPS only
- Implement rate limiting
- Monitor for suspicious activity

### **3. Application Security**
- Keep dependencies updated
- Implement proper error handling
- Use secure headers
- Validate all inputs

## 🚨 Troubleshooting

### **Common Issues**

1. **Services not starting**
   ```bash
   # Check logs
   docker-compose -f docker-compose.prod.yml logs
   
   # Check disk space
   df -h
   
   # Check memory
   free -h
   ```

2. **Webhook not receiving messages**
   - Verify webhook URLs in Twilio console
   - Check SSL certificate validity
   - Test webhook endpoint manually

3. **High memory usage**
   - Check for memory leaks
   - Increase server resources
   - Optimize ML model loading

4. **Database connection issues**
   - Check PostgreSQL logs
   - Verify connection string
   - Check network connectivity

### **Log Analysis**
```bash
# View application logs
tail -f logs/healthbot.log

# View error logs
grep "ERROR" logs/healthbot.log

# View access logs
tail -f logs/access.log
```

## 📈 Scaling

### **Horizontal Scaling**
- Use load balancer (nginx/HAProxy)
- Deploy multiple instances
- Use Redis for session sharing
- Implement database clustering

### **Vertical Scaling**
- Increase server resources
- Optimize ML model
- Use faster storage (SSD)
- Implement caching strategies

## 🎯 Success Metrics

### **Key Performance Indicators**
- **Uptime**: > 99.9%
- **Response Time**: < 2 seconds
- **Error Rate**: < 1%
- **User Satisfaction**: > 4.5/5

### **Monitoring Dashboard**
- Real-time metrics
- Error tracking
- User analytics
- Performance graphs

## 🆘 Support

### **Getting Help**
1. Check logs for errors
2. Review this documentation
3. Check GitHub issues
4. Contact support team

### **Emergency Procedures**
1. **Service Down**: Restart services
2. **High Load**: Scale up resources
3. **Security Breach**: Isolate and investigate
4. **Data Loss**: Restore from backup

---

## 🎉 Congratulations!

Your HealthBot AI Chatbot is now deployed and ready to help users with their health queries via WhatsApp and SMS!

**Next Steps:**
1. Monitor the system closely for the first few days
2. Set up automated backups
3. Configure alerting for critical issues
4. Plan for scaling as usage grows
5. Gather user feedback and iterate

**Remember**: This is a medical information tool, not a replacement for professional medical advice. Always include appropriate disclaimers and encourage users to consult healthcare professionals for serious concerns.