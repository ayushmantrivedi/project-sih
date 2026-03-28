from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from chatbot import generate_bot_response

app = Flask(__name__)

@app.route('/whatsapp', methods=['POST'])
def whatsapp():
    incoming_msg = request.values.get('Body', '').strip()
    reply_json = generate_bot_response(incoming_msg)
    reply = str(reply_json)
    resp = MessagingResponse()
    resp.message(reply)
    return str(resp)