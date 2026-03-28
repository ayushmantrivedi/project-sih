from flask import Flask, request, jsonify
from chatbot import generate_bot_response

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message = data.get("message", "")
    reply = generate_bot_response(user_message)
    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(port=5000, debug=True)