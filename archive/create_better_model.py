#!/usr/bin/env python3
"""
Create a better model with common diseases and symptoms
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re

def create_common_diseases_dataset():
    """Create a dataset with common diseases and their symptoms"""
    
    # Common diseases with their typical symptoms
    common_diseases = {
        "Common Cold": [
            "runny nose sneezing cough mild fever",
            "sore throat nasal congestion headache",
            "cough sneezing runny nose fatigue",
            "nasal congestion mild fever sore throat"
        ],
        "Flu": [
            "high fever body aches chills fatigue",
            "fever headache muscle pain weakness",
            "severe fatigue high fever cough",
            "body aches chills fever headache"
        ],
        "Hypertension": [
            "high blood pressure headache dizziness",
            "chest pain shortness of breath fatigue",
            "headache blurred vision nosebleeds",
            "dizziness chest pain shortness of breath"
        ],
        "Diabetes": [
            "frequent urination excessive thirst fatigue",
            "increased hunger weight loss blurry vision",
            "thirst frequent urination fatigue",
            "blurred vision slow healing wounds"
        ],
        "Asthma": [
            "wheezing shortness of breath chest tightness",
            "coughing wheezing difficulty breathing",
            "chest tightness shortness of breath wheezing",
            "difficulty breathing coughing wheezing"
        ],
        "Migraine": [
            "severe headache nausea light sensitivity",
            "throbbing headache nausea vomiting",
            "headache sensitivity to light sound",
            "severe head pain nausea light sensitivity"
        ],
        "Gastroenteritis": [
            "diarrhea vomiting nausea stomach cramps",
            "nausea vomiting diarrhea abdominal pain",
            "stomach pain diarrhea vomiting",
            "abdominal cramps diarrhea nausea"
        ],
        "Pneumonia": [
            "cough fever chest pain shortness of breath",
            "high fever cough chest pain",
            "shortness of breath cough fever",
            "chest pain cough difficulty breathing"
        ],
        "Allergies": [
            "sneezing runny nose itchy eyes",
            "itchy skin rash sneezing",
            "nasal congestion sneezing itchy eyes",
            "runny nose sneezing watery eyes"
        ],
        "Anxiety": [
            "worry restlessness difficulty concentrating",
            "panic attacks rapid heartbeat sweating",
            "nervousness tension muscle tension",
            "restlessness irritability sleep problems"
        ]
    }
    
    # Create dataset
    data = []
    for disease, symptoms_list in common_diseases.items():
        for symptoms in symptoms_list:
            data.append({
                'symptoms': symptoms,
                'disease': disease
            })
    
    # Add some variations
    variations = {
        "Common Cold": ["cold symptoms", "nasal congestion", "sore throat", "mild fever"],
        "Flu": ["influenza symptoms", "high fever", "body aches", "chills"],
        "Hypertension": ["high blood pressure", "chest pain", "headache", "dizziness"],
        "Diabetes": ["diabetes symptoms", "thirst", "frequent urination", "fatigue"],
        "Asthma": ["breathing problems", "wheezing", "chest tightness", "cough"],
        "Migraine": ["severe headache", "head pain", "nausea", "light sensitivity"],
        "Gastroenteritis": ["stomach flu", "diarrhea", "vomiting", "nausea"],
        "Pneumonia": ["lung infection", "chest pain", "cough", "fever"],
        "Allergies": ["allergic reaction", "sneezing", "itchy eyes", "runny nose"],
        "Anxiety": ["anxiety symptoms", "worry", "panic", "restlessness"]
    }
    
    for disease, symptom_variations in variations.items():
        for symptom in symptom_variations:
            data.append({
                'symptoms': symptom,
                'disease': disease
            })
    
    return pd.DataFrame(data)

def clean_text(text):
    """Clean text for processing"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def create_better_model():
    """Create a better model with common diseases"""
    print("🚀 Creating Better Model with Common Diseases")
    print("=" * 50)
    
    # Create dataset
    print("📊 Creating dataset with common diseases...")
    df = create_common_diseases_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"Unique diseases: {df['disease'].nunique()}")
    print(f"Diseases: {list(df['disease'].unique())}")
    
    # Clean data
    print("\n🧹 Cleaning data...")
    df['symptoms'] = df['symptoms'].apply(clean_text)
    
    # Prepare features
    X_text = df['symptoms'].values
    y = df['disease'].values
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Vectorize text
    print("📝 Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X_vectorized = vectorizer.fit_transform(X_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train model
    print("🌲 Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\n✅ Model training complete!")
    print(f"Train accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    # Save model
    model_bundle = {
        'model': model,
        'vectorizer': vectorizer,
        'label_encoder': le,
        'text_column': 'symptoms',
        'label_column': 'disease'
    }
    
    joblib.dump(model_bundle, 'better_health_model.joblib')
    print("💾 Better model saved as 'better_health_model.joblib'")
    
    # Test predictions
    print(f"\n🧪 Testing the better model:")
    test_cases = [
        "chest pain",
        "fever and cough",
        "headache and nausea",
        "shortness of breath",
        "diarrhea and vomiting",
        "high blood pressure"
    ]
    
    for test_text in test_cases:
        X_test_vec = vectorizer.transform([test_text])
        pred = model.predict(X_test_vec)[0]
        pred_label = le.inverse_transform([pred])[0]
        confidence = model.predict_proba(X_test_vec)[0].max()
        
        print(f"   '{test_text}' -> {pred_label} (confidence: {confidence:.1%})")
    
    return True

if __name__ == "__main__":
    try:
        success = create_better_model()
        if success:
            print("\n🎉 Better model created successfully!")
            print("This model focuses on common diseases and should give better predictions.")
        else:
            print("❌ Model creation failed")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
