#!/usr/bin/env python3
"""
Quick Start Script for GPU-Optimized Training
This script provides an easy way to start training with optimal settings
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check if required files exist
    required_files = [
        "train_gpu_optimized.py",
        "train_fast_gpu.py",
        "models/gpu_optimized_predict.py"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Required file not found: {file}")
            return False
    
    print("✅ All requirements met")
    return True

def check_gpu():
    """Check GPU availability"""
    print("🔍 Checking GPU availability...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if not gpus:
            print("⚠️  No GPU detected. Training will use CPU (slower)")
            return False
        
        print(f"✅ Found {len(gpus)} GPU(s)")
        return True
        
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False

def create_sample_data():
    """Create sample data if none exists"""
    print("🔄 Creating sample data...")
    
    if os.path.exists("sample_health_data.csv"):
        print("✅ Sample data already exists")
        return True
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample health data
        diseases = [
            "Common Cold", "Flu", "Hypertension", "Diabetes", "Asthma",
            "Migraine", "Gastroenteritis", "Pneumonia", "Allergies", "Anxiety"
        ]
        
        symptoms = [
            "fever and cough with headache",
            "high fever body aches chills",
            "chest pain shortness of breath",
            "frequent urination excessive thirst",
            "wheezing shortness of breath",
            "severe headache nausea",
            "diarrhea vomiting nausea",
            "cough fever chest pain",
            "sneezing runny nose itchy eyes",
            "worry restlessness difficulty concentrating"
        ]
        
        # Create more samples
        data = []
        for _ in range(1000):
            disease = np.random.choice(diseases)
            symptom = np.random.choice(symptoms)
            
            # Add some variation
            if np.random.random() > 0.5:
                symptom += " and fatigue"
            
            data.append({
                'symptoms': symptom,
                'disease': disease,
                'age': np.random.randint(18, 80),
                'severity': np.random.randint(1, 5)
            })
        
        df = pd.DataFrame(data)
        df.to_csv("sample_health_data.csv", index=False)
        
        print("✅ Sample data created: sample_health_data.csv")
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return False

def run_training_script(script_name, args):
    """Run a training script with given arguments"""
    print(f"🚀 Running {script_name}...")
    
    cmd = [sys.executable, script_name] + args
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Quick Start GPU Training')
    parser.add_argument('--mode', choices=['standard', 'ultra', 'memory'], default='standard',
                       help='Training mode (default: standard)')
    parser.add_argument('--csv', type=str, default='sample_health_data.csv',
                       help='Path to CSV data file')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Sample size for large datasets')
    parser.add_argument('--test_only', action='store_true',
                       help='Only run tests, do not train')
    
    args = parser.parse_args()
    
    print("🚀 Quick Start GPU Training")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"CSV: {args.csv}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Check requirements
    if not check_requirements():
        print("❌ Requirements not met. Please install required packages.")
        return False
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Create sample data if needed
    if not os.path.exists(args.csv):
        if not create_sample_data():
            print("❌ Failed to create sample data")
            return False
    
    # Run tests
    print("\n🧪 Running system tests...")
    test_result = subprocess.run([sys.executable, "test_gpu_optimization.py"], 
                                capture_output=True, text=True)
    
    if test_result.returncode != 0:
        print("⚠️  Some tests failed, but continuing with training...")
        print("Test output:", test_result.stdout)
    else:
        print("✅ All tests passed")
    
    if args.test_only:
        print("🧪 Test-only mode. Training skipped.")
        return True
    
    # Prepare training arguments
    training_args = [
        '--csv', args.csv,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size)
    ]
    
    if args.sample_size:
        training_args.extend(['--sample_size', str(args.sample_size)])
    
    # Choose training script based on mode
    if args.mode == 'standard':
        script_name = "train_gpu_optimized.py"
        training_args.extend(['--output_dir', 'gpu_optimized_model'])
    elif args.mode == 'ultra':
        script_name = "train_fast_gpu.py"
        training_args.extend(['--output_dir', 'ultra_optimized_model'])
    elif args.mode == 'memory':
        script_name = "train_large_memory_optimized.py"
        if not args.sample_size:
            training_args.extend(['--sample_size', '50000'])
    
    # Run training
    print(f"\n🚀 Starting {args.mode} training...")
    print(f"Script: {script_name}")
    print(f"Arguments: {' '.join(training_args)}")
    print()
    
    start_time = time.time()
    success = run_training_script(script_name, training_args)
    training_time = time.time() - start_time
    
    if success:
        print(f"\n🎉 Training completed successfully in {training_time:.2f} seconds!")
        print("\n📦 Model artifacts saved:")
        
        if args.mode == 'standard':
            print("   - gpu_optimized_model/")
        elif args.mode == 'ultra':
            print("   - ultra_optimized_model/")
        elif args.mode == 'memory':
            print("   - Model artifacts in current directory")
        
        print("\n🎯 Your model is ready for use!")
        print("\n💡 Next steps:")
        print("   1. Test your model with: python models/gpu_optimized_predict.py")
        print("   2. Integrate with your application")
        print("   3. Monitor performance and accuracy")
        
    else:
        print(f"\n❌ Training failed after {training_time:.2f} seconds")
        print("\n💡 Troubleshooting tips:")
        print("   1. Check GPU availability and drivers")
        print("   2. Reduce batch size or sample size")
        print("   3. Check available memory")
        print("   4. Review error messages above")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)