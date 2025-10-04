#!/usr/bin/env python3
"""
Quick model training script for your synthetic dataset
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re

def clean_text(text):
    """Simple text cleaning"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def quick_train():
    """Quick training using TF-IDF + Random Forest"""
    print("🚀 Starting Quick Model Training...")
    
    # Load your dataset
    print("📊 Loading dataset...")
    df = pd.read_csv("augmented_synthetic_health_dataset.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Find text and label columns
    text_col = None
    label_col = None
    
    # Prefer 'normalized_symptoms' for text and 'disease' for labels
    if 'normalized_symptoms' in df.columns:
        text_col = 'normalized_symptoms'
    elif 'symptoms' in df.columns:
        text_col = 'symptoms'
    
    if 'disease' in df.columns:
        label_col = 'disease'
    elif 'disease_code' in df.columns:
        label_col = 'disease_code'
    
    if text_col is None:
        # Use first text-like column
        text_col = df.select_dtypes(include=['object']).columns[0]
    if label_col is None:
        # Use last column as label
        label_col = df.columns[-1]
    
    print(f"Using text column: {text_col}")
    print(f"Using label column: {label_col}")
    
    # Clean and prepare data
    print("🧹 Cleaning data...")
    df[text_col] = df[text_col].apply(clean_text)
    
    # Remove empty rows
    df = df[df[text_col].str.len() > 0]
    df = df[df[label_col].notna()]
    
    print(f"Clean dataset shape: {df.shape}")
    print(f"Unique labels: {df[label_col].nunique()}")
    print(f"Label distribution:\n{df[label_col].value_counts()}")
    
    # Prepare features
    X_text = df[text_col].values
    y = df[label_col].values
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Vectorize text
    print("📝 Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
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
    
    print(f"✅ Training complete!")
    print(f"Train accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    # Save model
    model_bundle = {
        'model': model,
        'vectorizer': vectorizer,
        'label_encoder': le,
        'text_column': text_col,
        'label_column': label_col
    }
    
    joblib.dump(model_bundle, 'quick_trained_model.joblib')
    print("💾 Model saved as 'quick_trained_model.joblib'")
    
    # Test prediction
    test_text = "fever and cough with headache"
    X_test_vec = vectorizer.transform([test_text])
    pred = model.predict(X_test_vec)[0]
    pred_label = le.inverse_transform([pred])[0]
    confidence = model.predict_proba(X_test_vec)[0].max()
    
    print(f"🧪 Test prediction: '{test_text}' -> {pred_label} (confidence: {confidence:.3f})")
    
    return True

if __name__ == "__main__":
    try:
        success = quick_train()
        if success:
            print("\n🎉 Quick training completed successfully!")
            print("Your model is ready to use!")
        else:
            print("❌ Training failed")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
