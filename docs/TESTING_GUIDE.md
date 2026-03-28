# 🧪 HealthBot AI Chatbot - Testing Guide

## 📋 **Quick Testing Steps for Your System**

Since you have Python dependencies installed, follow these steps to test your chatbot:

### **Step 1: Basic Import Test (30 seconds)**
```bash
# Test if all modules can be imported
python3 -c "
import flask, joblib, numpy, pandas, sklearn, tensorflow, transformers, requests, twilio
print('✅ All dependencies imported successfully!')
"
```

### **Step 2: Test ML Model (1 minute)**
```bash
# Test model loading and prediction
python3 -c "
import joblib
model = joblib.load('quick_trained_model.joblib')
print('✅ Model loaded successfully!')
print(f'Model type: {type(model)}')
"
```

### **Step 3: Test Disease Prediction (1 minute)**
```bash
# Test the predict_disease function
python3 -c "
from models.ml_predict import predict_disease
prediction, confidence, probabilities = predict_disease('I have fever and cough')
print(f'✅ Prediction: {prediction}')
print(f'✅ Confidence: {confidence:.3f}')
"
```

### **Step 4: Test Chatbot Responses (2 minutes)**
```bash
# Test different types of queries
python3 -c "
from chatbot import generate_bot_response

# Test symptom analysis
response = generate_bot_response('I have fever and cough')
print('Symptom Analysis:', response)

# Test COVID data
response = generate_bot_response('Show me COVID statistics')
print('COVID Data:', response)

# Test clinical trials
response = generate_bot_response('Find clinical trials for diabetes')
print('Clinical Trials:', response)
"
```

### **Step 5: Test API Integrations (2 minutes)**
```bash
# Test each API
python3 -c "
from apis.clinicaltrials import get_clinical_trials
from apis.cowin import get_cowin_stats
from apis.mohfw import get_mohfw_data
from apis.umls import get_umls_info

print('Testing Clinical Trials API...')
trials = get_clinical_trials('diabetes')
print(f'✅ Clinical Trials: {len(trials)} results')

print('Testing CoWIN API...')
cowin = get_cowin_stats()
print(f'✅ CoWIN: {type(cowin)}')

print('Testing MoHFW API...')
mohfw = get_mohfw_data()
print(f'✅ MoHFW: {type(mohfw)}')

print('Testing UMLS API...')
umls = get_umls_info('hypertension')
print(f'✅ UMLS: {type(umls)}')
"
```

### **Step 6: Test Web Server (2 minutes)**
```bash
# Test Flask app startup
python3 -c "
from main import app
print('✅ Flask app imported successfully')
print(f'App name: {app.name}')
print(f'Debug mode: {app.debug}')

# Test routes
routes = [rule.rule for rule in app.url_map.iter_rules()]
print(f'Available routes: {routes}')
"
```

### **Step 7: Run Web Server Test (3 minutes)**
```bash
# Start the web server in background
python3 main.py &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Test health endpoint
curl http://localhost:5000/health

# Test chat endpoint
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I have fever and cough", "user_id": "test_user"}'

# Stop the server
kill $SERVER_PID
```

### **Step 8: Comprehensive Test (5 minutes)**
```bash
# Run the comprehensive test script
python3 test_your_system.py
```

---

## 🚀 **Quick Start Commands**

### **Start the Chatbot Server:**
```bash
python3 main.py
```

### **Test in Another Terminal:**
```bash
# Health check
curl http://localhost:5000/health

# Chat with the bot
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I have fever and cough", "user_id": "test_user"}'

# Test different queries
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me COVID statistics", "user_id": "test_user"}'

curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Find clinical trials for diabetes", "user_id": "test_user"}'
```

---

## 📱 **Test WhatsApp and SMS Webhooks**

### **Test WhatsApp Webhook:**
```bash
# Simulate WhatsApp webhook call
curl -X POST http://localhost:5000/webhook/whatsapp \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "Body=I have fever and cough&From=whatsapp:+1234567890"
```

### **Test SMS Webhook:**
```bash
# Simulate SMS webhook call
curl -X POST http://localhost:5000/webhook/sms \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "Body=I have fever and cough&From=+1234567890"
```

---

## 🔧 **Expected Results**

### **✅ Successful Test Results:**
- All imports work without errors
- Model loads successfully (5.9MB file)
- Disease predictions return valid results
- Chatbot responds to different query types
- APIs return data from health services
- Web server starts and responds to requests
- Webhooks process messages correctly

### **⚠️ Common Issues:**
- **Import errors**: Install missing packages with `pip install -r requirements.txt`
- **Model loading errors**: Check if `quick_trained_model.joblib` exists
- **API errors**: Check internet connection for external API calls
- **Webhook errors**: Ensure Twilio credentials are configured

---

## 🎯 **What Each Test Validates**

1. **Import Test**: Verifies all Python dependencies are installed
2. **Model Test**: Confirms ML model can be loaded and used
3. **Prediction Test**: Validates disease prediction functionality
4. **Chatbot Test**: Tests the main chatbot logic and responses
5. **API Test**: Verifies external health API integrations
6. **Web Server Test**: Confirms Flask app and routes work
7. **Webhook Test**: Tests WhatsApp and SMS message processing

---

## 🚀 **Ready for Production?**

If all tests pass, your chatbot is ready for:
- ✅ Local testing and development
- ✅ Production deployment
- ✅ WhatsApp integration (with Twilio setup)
- ✅ SMS integration (with Twilio setup)
- ✅ Web API usage
- ✅ Scaling to multiple users

**Next Steps:**
1. Set up Twilio account and credentials
2. Configure webhook URLs
3. Deploy to production server
4. Start helping users with their health queries!

---

## 🆘 **Need Help?**

If any tests fail:
1. Check the error messages carefully
2. Install missing dependencies: `pip install -r requirements.txt`
3. Verify all files are in the correct locations
4. Check your Python version (3.9+ recommended)
5. Review the error logs for specific issues

Your HealthBot AI Chatbot is professionally built and ready to help users with their health queries! 🏥🤖