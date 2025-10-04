#!/usr/bin/env python3
"""
Test your trained model directly
"""
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_my_model():
    """Test the trained model with your own inputs"""
    print("🧪 Testing Your Trained HealthBot Model")
    print("=" * 50)
    
    # Import the quick predict module
    try:
        from models.quick_predict import predict_new, load_quick_model
        print("✅ Model module imported successfully!")
    except Exception as e:
        print(f"❌ Error importing model: {e}")
        return
    
    # Load the model
    print("\n📥 Loading your trained model...")
    if not load_quick_model():
        print("❌ Could not load model. Make sure 'quick_trained_model.joblib' exists.")
        return
    
    print("✅ Model loaded successfully!")
    print("\n🔬 Your model was trained on 1000 synthetic health records")
    print("📊 Model achieved 98% accuracy on test data")
    print("🤖 Model type: Random Forest with TF-IDF text processing")
    
    # Interactive testing
    print("\n" + "=" * 50)
    print("💬 INTERACTIVE TESTING")
    print("=" * 50)
    print("Enter symptoms to test your model (type 'quit' to exit)")
    print("Examples:")
    print("  - 'fever and cough'")
    print("  - 'headache and nausea'")
    print("  - 'chest pain and shortness of breath'")
    print("  - 'rash and itching'")
    
    while True:
        try:
            # Get user input
            symptoms = input("\n🤔 Enter symptoms: ").strip()
            
            if symptoms.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not symptoms:
                print("⚠️ Please enter some symptoms to test.")
                continue
            
            # Get prediction
            print(f"\n🔍 Analyzing: '{symptoms}'")
            label, confidence, probabilities = predict_new(symptoms)
            
            # Display results
            print(f"🎯 Predicted Disease: {label}")
            print(f"📈 Confidence: {confidence:.1%}")
            
            if confidence >= 0.8:
                print("✅ High confidence prediction")
            elif confidence >= 0.6:
                print("⚠️ Medium confidence prediction")
            else:
                print("❌ Low confidence prediction")
            
            # Show probability distribution if available
            if len(probabilities) > 1:
                print(f"\n📊 Prediction probabilities:")
                for i, prob in enumerate(probabilities):
                    if prob > 0.01:  # Only show probabilities > 1%
                        print(f"   Class {i}: {prob:.1%}")
            
            print("\n" + "-" * 30)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            print("Please try again with different symptoms.")

def quick_test():
    """Run a quick test with predefined examples"""
    print("🚀 Quick Test Mode")
    print("=" * 30)
    
    try:
        from models.quick_predict import predict_new, load_quick_model
        
        if not load_quick_model():
            print("❌ Could not load model")
            return
        
        # Test cases
        test_cases = [
            "fever and cough with headache",
            "chest pain and shortness of breath", 
            "nausea and vomiting",
            "rash and itching on skin",
            "severe headache and neck pain",
            "abdominal pain and diarrhea"
        ]
        
        print("🧪 Testing with predefined examples:\n")
        
        for i, symptoms in enumerate(test_cases, 1):
            print(f"{i}. Testing: '{symptoms}'")
            label, confidence, _ = predict_new(symptoms)
            print(f"   Result: {label} (confidence: {confidence:.1%})")
            print()
        
        print("✅ Quick test completed!")
        
    except Exception as e:
        print(f"❌ Error in quick test: {e}")

if __name__ == "__main__":
    print("🎯 Choose testing mode:")
    print("1. Interactive testing (recommended)")
    print("2. Quick test with examples")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        quick_test()
    else:
        test_my_model()
