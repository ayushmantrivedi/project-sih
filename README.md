# HealthBot AI Chatbot 🏥🤖

A comprehensive AI-powered health chatbot that provides disease prediction, medical information, and health data through multiple channels including WhatsApp, SMS, and web API.

## 🌟 Features

### Core Capabilities
- **AI Disease Prediction**: Advanced ML model using Bio-ClinicalBERT for symptom analysis
- **Multi-language Support**: English and Hindi language support
- **Confidence-based Responses**: Smart confidence thresholds with appropriate disclaimers
- **Real-time Health Data**: Integration with multiple health APIs

### API Integrations
- **ClinicalTrials.gov**: Search for relevant clinical trials
- **CoWIN**: Vaccination statistics and information
- **MoHFW**: COVID-19 data and statistics
- **UMLS**: Medical terminology and definitions
- **Infermedica**: Advanced symptom analysis (optional)

### Communication Channels
- **Web API**: RESTful API for web applications
- **WhatsApp**: Full WhatsApp Business API integration
- **SMS**: SMS support via Twilio
- **Rich Media**: Support for text, images, and documents

### Production Features
- **Rate Limiting**: Configurable rate limiting for API abuse prevention
- **Logging & Monitoring**: Comprehensive logging and analytics
- **Error Handling**: Robust error handling and fallback responses
- **Docker Support**: Complete containerization for easy deployment
- **Database Integration**: SQLite/PostgreSQL support with Redis caching

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker (optional)
- Twilio account (for WhatsApp/SMS)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sihdemo
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

4. **Train the ML model** (if not already trained)
   ```bash
   python models/sihdemo.py --csv augmented_synthetic_health_dataset.csv
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

2. **Check health**
   ```bash
   curl http://localhost:5000/health
   ```

## 📁 Project Structure

```
sihdemo/
├── apis/                    # API integrations
│   ├── clinicaltrials.py   # ClinicalTrials.gov API
│   ├── cowin.py            # CoWIN vaccination API
│   ├── mohfw.py            # MoHFW COVID-19 API
│   ├── umls.py             # UMLS medical terminology
│   └── infermedica_api.py  # Infermedica symptom analysis
├── config/                 # Configuration management
│   ├── settings.py         # Main configuration
│   └── __init__.py
├── models/                 # ML models and prediction
│   ├── sihdemo.py          # Main ML model training
│   ├── ml_predict.py       # Disease prediction interface
│   └── __init__.py
├── utils/                  # Utility functions
│   ├── logger.py           # Logging configuration
│   └── __init__.py
├── database/               # Database files
├── logs/                   # Application logs
├── tests/                  # Test files
├── docs/                   # Documentation
├── deployment/             # Deployment configurations
├── main.py                 # Main application entry point
├── chatbot.py              # Chatbot logic
├── app.py                  # Legacy Flask app
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose setup
└── README.md              # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `env.example`:

```env
# Application Settings
DEBUG=False
SECRET_KEY=your-secret-key
HOST=0.0.0.0
PORT=5000

# Twilio Configuration (Required for WhatsApp/SMS)
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
TWILIO_SMS_NUMBER=+1234567890

# API Keys (Optional)
UMLS_API_KEY=your_umls_api_key
INFERMEDICA_APP_ID=your_infermedica_app_id
INFERMEDICA_APP_KEY=your_infermedica_app_key
```

### Model Configuration

The ML model can be configured in `config/settings.py`:

```python
@dataclass
class MLConfig:
    MODEL_NAME: str = "emilyalsentzer/Bio_ClinicalBERT"
    MAX_LENGTH: int = 64
    BATCH_SIZE: int = 16
    HIGH_CONFIDENCE_THRESHOLD: float = 0.8
    MEDIUM_CONFIDENCE_THRESHOLD: float = 0.6
    LOW_CONFIDENCE_THRESHOLD: float = 0.4
```

## 📱 API Usage

### Web API

**Chat Endpoint**
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I have fever and cough", "user_id": "user123"}'
```

**Health Check**
```bash
curl http://localhost:5000/health
```

### WhatsApp Integration

1. Set up Twilio WhatsApp Sandbox
2. Configure webhook URL: `https://yourdomain.com/webhook/whatsapp`
3. Send messages to your WhatsApp number

### SMS Integration

1. Configure Twilio SMS number
2. Set webhook URL: `https://yourdomain.com/webhook/sms`
3. Send SMS to your Twilio number

## 🤖 Chatbot Responses

The chatbot can respond to various types of queries:

### Symptom Analysis
```
User: "I have fever and headache"
Bot: {
  "type": "diagnosis",
  "disease": "Common Cold",
  "confidence": 0.85,
  "confidence_level": "high",
  "probabilities": [...]
}
```

### API Queries
```
User: "Show me COVID statistics"
Bot: {
  "type": "covid",
  "content": [...]
}
```

### Medical Definitions
```
User: "Define hypertension"
Bot: {
  "type": "umls",
  "content": {...}
}
```

## 🔍 Monitoring & Logging

### Log Files
- Application logs: `logs/healthbot.log`
- User interactions: Logged with timestamps and confidence scores
- API calls: Monitored with response times and error tracking

### Health Monitoring
- Health check endpoint: `/health`
- Docker health checks configured
- Rate limiting monitoring

## 🚀 Deployment

### Production Deployment

1. **Set up production environment**
   ```bash
   export DEBUG=False
   export SECRET_KEY=your-production-secret-key
   ```

2. **Use production WSGI server**
   ```bash
   gunicorn --bind 0.0.0.0:5000 --workers 4 main:app
   ```

3. **Set up reverse proxy** (nginx recommended)
4. **Configure SSL certificates**
5. **Set up monitoring and alerting**

### Cloud Deployment

The application is ready for deployment on:
- AWS (ECS, EC2, Lambda)
- Google Cloud (Cloud Run, Compute Engine)
- Azure (Container Instances, App Service)
- Heroku
- DigitalOcean

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Test specific components
pytest tests/test_chatbot.py
pytest tests/test_apis.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation in `/docs`
- Review the logs in `/logs`

## 🔮 Roadmap

- [ ] Multi-language model support
- [ ] Voice message processing
- [ ] Appointment booking integration
- [ ] Electronic health records (EHR) integration
- [ ] Advanced analytics dashboard
- [ ] Mobile app development
- [ ] Integration with more health APIs
- [ ] Machine learning model improvements

---

**Disclaimer**: This chatbot is for informational purposes only and should not replace professional medical advice. Always consult with qualified healthcare professionals for medical concerns.