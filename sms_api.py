from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

@app.route('/sms', methods=['POST'])
def sms():
    incoming_msg = request.values.get('Body', '').strip()
    resp = MessagingResponse()
    # Call your chatbot logic here
    reply = "Chatbot response here"
    resp.message(reply)
    return str(resp)