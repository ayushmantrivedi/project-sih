from .sihdemo import predict_new

def predict_disease(symptom_text):
    # No numeric features here, but you can expand
    label, conf, probs = predict_new(symptom_text, None)
    return label, conf, probs