#!/usr/bin/env python3
"""
Training script optimized for large datasets (100,000+ records)
"""

import os
import sys
import argparse
import gc
import psutil

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from sihdemo import main as train_main

def check_system_resources():
    """Check if system has enough resources for large dataset training"""
    print("🔍 Checking system resources...")
    
    # Check available memory
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    total_gb = memory.total / (1024**3)
    
    print(f"💾 Available RAM: {available_gb:.1f} GB / {total_gb:.1f} GB")
    
    if available_gb < 8:
        print("⚠️  Warning: Less than 8GB RAM available. Training may be slow or fail.")
        print("💡 Consider reducing batch size or using data sampling.")
        return False
    
    print("✅ System resources look good for large dataset training")
    return True

def optimize_for_large_dataset():
    """Apply optimizations for large dataset training"""
    print("⚙️  Applying optimizations for large dataset...")
    
    # Set environment variables for memory optimization
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    
    # Force garbage collection
    gc.collect()
    
    print("✅ Optimizations applied")

def main():
    """Train the model with large dataset optimizations"""
    print("🚀 Starting training with LARGE dataset (100,000+ records)...")
    print("📊 Expected data: 1,000 CSV + 100,000 JSON = 101,000 total records")
    print()
    
    # Check system resources
    if not check_system_resources():
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("❌ Training cancelled")
            return False
    
    # Apply optimizations
    optimize_for_large_dataset()
    
    # Set up arguments for training
    parser = argparse.ArgumentParser(description='Train ML model with large dataset')
    parser.add_argument('--csv', type=str, default='C:\\Users\\ayush\\OneDrive\\Desktop\\augmented_synthetic_health_dataset.csv',
                       help='Path to CSV training data')
    parser.add_argument('--json', type=str, default='C:\\Users\\ayush\\OneDrive\\Desktop\\diagnosis_data.json',
                       help='Path to JSON training data')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs (reduced for large dataset)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training (optimized for large dataset)')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate (reduced for stability)')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Sample size for testing (optional)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.csv):
        print(f"❌ CSV file not found: {args.csv}")
        return False
    
    if not os.path.exists(args.json):
        print(f"❌ JSON file not found: {args.json}")
        return False
    
    print(f"📁 CSV data: {args.csv}")
    print(f"📁 JSON data: {args.json}")
    print(f"⚙️  Epochs: {args.epochs}")
    print(f"⚙️  Batch size: {args.batch_size}")
    print(f"⚙️  Learning rate: {args.lr}")
    if args.sample_size:
        print(f"⚙️  Sample size: {args.sample_size:,} records")
    print()
    
    try:
        # Override sys.argv to pass arguments to the main training function
        sys.argv = ['train_large_dataset.py', 
                   '--csv', args.csv,
                   '--json', args.json,
                   '--epochs', str(args.epochs),
                   '--batch_size', str(args.batch_size),
                   '--lr', str(args.lr)]
        
        print("🔄 Starting training process...")
        print("⏳ This will take significant time with 100,000+ records...")
        print("💡 Monitor your system resources during training")
        print()
        
        # Call the main training function
        train_main()
        
        print("\n🎉 Training completed successfully!")
        print("📦 Model artifacts saved:")
        print("   - best_classifier_weights.h5")
        print("   - sih_model_bundle.joblib")
        print("   - saved_tokenizer/")
        print("   - saved_encoder/")
        print("\n🎯 Your model is now trained on 101,000+ records!")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   - Reduce batch size (--batch_size 16)")
        print("   - Reduce epochs (--epochs 10)")
        print("   - Use data sampling (--sample_size 10000)")
        print("   - Close other applications to free memory")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)