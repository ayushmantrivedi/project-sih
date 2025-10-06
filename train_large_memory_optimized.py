#!/usr/bin/env python3
"""
Memory-optimized training script for very large datasets (100,000+ records)
Uses data sampling and reduced cross-validation to prevent memory errors
"""

import os
import sys
import argparse
import gc
import psutil
import pandas as pd
import numpy as np

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

def check_memory():
    """Check available memory"""
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    total_gb = memory.total / (1024**3)
    print(f"💾 Available RAM: {available_gb:.1f} GB / {total_gb:.1f} GB")
    return available_gb

def sample_large_dataset(csv_path, json_path, sample_size=50000):
    """Sample a subset of the large dataset for training"""
    print(f"🔄 Sampling {sample_size:,} records from large dataset...")
    
    # Load and sample CSV data
    df_csv = pd.read_csv(csv_path)
    if len(df_csv) > sample_size // 2:
        df_csv = df_csv.sample(n=sample_size // 2, random_state=42)
    print(f"✅ CSV sample: {len(df_csv):,} records")
    
    # Load and sample JSON data
    df_json = pd.DataFrame()
    if os.path.exists(json_path):
        try:
            import json
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df_json = pd.DataFrame(data)
            
            # Sample JSON data
            if len(df_json) > sample_size // 2:
                df_json = df_json.sample(n=sample_size // 2, random_state=42)
            print(f"✅ JSON sample: {len(df_json):,} records")
        except Exception as e:
            print(f"⚠️  JSON loading failed: {e}")
    
    # Combine samples
    if not df_json.empty:
        # Find common columns
        common_cols = set(df_csv.columns) & set(df_json.columns)
        df_csv = df_csv[list(common_cols)]
        df_json = df_json[list(common_cols)]
        df_combined = pd.concat([df_csv, df_json], ignore_index=True)
    else:
        df_combined = df_csv
    
    print(f"✅ Combined sample: {len(df_combined):,} records")
    return df_combined

def create_sampled_files(df, output_csv="sampled_data.csv"):
    """Create sampled CSV file for training"""
    df.to_csv(output_csv, index=False)
    print(f"💾 Saved sampled data to: {output_csv}")
    return output_csv

def train_with_sampling():
    """Train model with data sampling to avoid memory issues"""
    print("🚀 Memory-Optimized Training for Large Dataset")
    print("=" * 60)
    
    # Check memory
    available_memory = check_memory()
    if available_memory < 4:
        print("⚠️  Low memory detected. Using aggressive sampling.")
        sample_size = 20000
    elif available_memory < 8:
        print("⚠️  Medium memory. Using moderate sampling.")
        sample_size = 40000
    else:
        print("✅ Good memory available. Using larger sample.")
        sample_size = 60000
    
    # Set up arguments
    parser = argparse.ArgumentParser(description='Memory-optimized training')
    parser.add_argument('--csv', type=str, default='C:\\Users\\ayush\\OneDrive\\Desktop\\augmented_synthetic_health_dataset.csv')
    parser.add_argument('--json', type=str, default='C:\\Users\\ayush\\OneDrive\\Desktop\\diagnosis_data.json')
    parser.add_argument('--sample_size', type=int, default=sample_size)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--folds', type=int, default=3)  # Reduced folds for memory
    
    args = parser.parse_args()
    
    print(f"📊 Sample size: {args.sample_size:,} records")
    print(f"⚙️  Epochs: {args.epochs}")
    print(f"⚙️  Batch size: {args.batch_size}")
    print(f"⚙️  Cross-validation folds: {args.folds}")
    print()
    
    try:
        # Sample the large dataset
        df_sampled = sample_large_dataset(args.csv, args.json, args.sample_size)
        
        # Create temporary sampled file
        sampled_file = create_sampled_files(df_sampled, "temp_sampled_data.csv")
        
        # Import and modify the training function
        from sihdemo import main as original_main
        
        # Override sys.argv for the training
        sys.argv = ['train_large_memory_optimized.py',
                   '--csv', sampled_file,
                   '--json', '',  # No JSON since we already combined
                   '--epochs', str(args.epochs),
                   '--batch_size', str(args.batch_size)]
        
        print("🔄 Starting training with sampled data...")
        print("⏳ This will be much faster and use less memory...")
        print()
        
        # Run training
        original_main()
        
        print("\n🎉 Training completed successfully!")
        print("📦 Model artifacts saved:")
        print("   - best_classifier_weights.h5")
        print("   - sih_model_bundle.joblib")
        print("   - saved_tokenizer/")
        print("   - saved_encoder/")
        
        # Clean up temporary file
        if os.path.exists(sampled_file):
            os.remove(sampled_file)
            print(f"🧹 Cleaned up temporary file: {sampled_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("\n💡 Try reducing sample size or batch size")
        return False

def main():
    """Main function"""
    success = train_with_sampling()
    
    if success:
        print("\n🎯 Training completed with memory optimization!")
        print("💡 The model was trained on a representative sample of your large dataset")
        print("💡 This approach prevents memory errors while maintaining good performance")
    else:
        print("\n❌ Training failed. Try reducing sample size further.")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()