#!/usr/bin/env python3
"""
Training script that demonstrates the integration of JSON data with the ML model
"""

import os
import sys
import argparse

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from sihdemo import main as train_main

def main():
    """Train the model with both CSV and JSON data"""
    print("🚀 Starting training with integrated CSV + JSON data...")
    
    # Set up arguments for training
    parser = argparse.ArgumentParser(description='Train ML model with CSV and JSON data')
    parser.add_argument('--csv', type=str, default='C:\\Users\\ayush\\OneDrive\\Desktop\\augmented_synthetic_health_dataset.csv',
                       help='Path to CSV training data')
    parser.add_argument('--json', type=str, default='/workspace/diagnosis_data.json',
                       help='Path to JSON training data')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs (reduced for demo)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.csv):
        print(f"❌ CSV file not found: {args.csv}")
        print("💡 Please ensure the CSV file exists or provide the correct path")
        return False
    
    if not os.path.exists(args.json):
        print(f"❌ JSON file not found: {args.json}")
        print("💡 Please ensure the JSON file exists or provide the correct path")
        return False
    
    print(f"📁 CSV data: {args.csv}")
    print(f"📁 JSON data: {args.json}")
    print(f"⚙️  Epochs: {args.epochs}")
    print(f"⚙️  Batch size: {args.batch_size}")
    print(f"⚙️  Learning rate: {args.lr}")
    print()
    
    try:
        # Override sys.argv to pass arguments to the main training function
        sys.argv = ['train_with_json.py', 
                   '--csv', args.csv,
                   '--json', args.json,
                   '--epochs', str(args.epochs),
                   '--batch_size', str(args.batch_size),
                   '--lr', str(args.lr)]
        
        # Call the main training function
        train_main()
        
        print("\n🎉 Training completed successfully!")
        print("📦 Model artifacts saved:")
        print("   - best_classifier_weights.h5")
        print("   - sih_model_bundle.joblib")
        print("   - saved_tokenizer/")
        print("   - saved_encoder/")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)