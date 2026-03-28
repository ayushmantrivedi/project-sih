#!/usr/bin/env python3
"""
Debug your model to see what's happening
"""
import joblib
import pandas as pd

def debug_model():
    """Debug the trained model"""
    print("🔍 Debugging Your Trained Model")
    print("=" * 50)
    
    # Load the model
    try:
        bundle = joblib.load('quick_trained_model.joblib')
        print("✅ Model loaded successfully!")
        
        # Check what's in the model
        print(f"\n📊 Model components:")
        print(f"   - Model type: {type(bundle['model'])}")
        print(f"   - Vectorizer type: {type(bundle['vectorizer'])}")
        print(f"   - Label encoder type: {type(bundle['label_encoder'])}")
        
        # Check the label encoder
        le = bundle['label_encoder']
        print(f"\n🏷️ Unique diseases in your model: {len(le.classes_)}")
        print("First 10 diseases:")
        for i, disease in enumerate(le.classes_[:10]):
            print(f"   {i}: {disease}")
        
        print(f"\nLast 10 diseases:")
        for i, disease in enumerate(le.classes_[-10:]):
            print(f"   {len(le.classes_)-10+i}: {disease}")
        
        # Test prediction
        print(f"\n🧪 Testing prediction:")
        test_text = "chest pain"
        vectorizer = bundle['vectorizer']
        model = bundle['model']
        
        # Vectorize text
        X_vec = vectorizer.transform([test_text])
        
        # Get prediction
        pred = model.predict(X_vec)[0]
        pred_label = le.inverse_transform([pred])[0]
        confidence = model.predict_proba(X_vec)[0].max()
        probabilities = model.predict_proba(X_vec)[0]
        
        print(f"   Input: '{test_text}'")
        print(f"   Predicted: {pred_label}")
        print(f"   Confidence: {confidence:.1%}")
        
        # Show top 5 predictions
        print(f"\n📈 Top 5 predictions:")
        sorted_indices = probabilities.argsort()[::-1]
        for i in range(min(5, len(sorted_indices))):
            idx = sorted_indices[i]
            disease = le.inverse_transform([idx])[0]
            prob = probabilities[idx]
            print(f"   {i+1}. {disease}: {prob:.1%}")
        
        # Check your original dataset
        print(f"\n📋 Checking your original dataset:")
        df = pd.read_csv("augmented_synthetic_health_dataset.csv")
        print(f"   Dataset shape: {df.shape}")
        print(f"   Unique diseases: {df['disease'].nunique()}")
        print(f"   First 10 diseases in dataset:")
        for i, disease in enumerate(df['disease'].unique()[:10]):
            print(f"   {i+1}. {disease}")
            
        print(f"\n💡 The issue:")
        print(f"   - Your model has {len(le.classes_)} unique diseases")
        print(f"   - Each disease appears only once in training")
        print(f"   - Model is trying to predict from 1000 different diseases")
        print(f"   - Many diseases are not relevant to common symptoms")
        print(f"   - This causes low confidence and irrelevant predictions")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_model()
