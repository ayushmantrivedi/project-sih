#!/usr/bin/env python3
"""
Standalone EDA and Preprocessing Tool
Use this to analyze and clean any medical dataset before training
"""

import os
import sys
import pandas as pd
import json
import argparse
from typing import Optional

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from sihdemo import perform_eda_and_preprocessing

def load_dataset(csv_path: Optional[str] = None, json_path: Optional[str] = None):
    """Load dataset from CSV and/or JSON files"""
    datasets = []
    
    if csv_path and os.path.exists(csv_path):
        print(f"📁 Loading CSV: {csv_path}")
        df_csv = pd.read_csv(csv_path)
        datasets.append(("CSV", df_csv))
    
    if json_path and os.path.exists(json_path):
        print(f"📁 Loading JSON: {json_path}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df_json = pd.DataFrame(data)
            datasets.append(("JSON", df_json))
        except Exception as e:
            print(f"❌ Error loading JSON: {e}")
    
    if not datasets:
        raise ValueError("No valid datasets found")
    
    if len(datasets) == 1:
        name, df = datasets[0]
        return df, name
    
    # Combine multiple datasets
    print("🔄 Combining datasets...")
    combined_df = pd.concat([df for _, df in datasets], ignore_index=True)
    return combined_df, "Combined"

def save_cleaned_dataset(df, output_path: str):
    """Save cleaned dataset to file"""
    if output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    elif output_path.endswith('.json'):
        df.to_json(output_path, orient='records', indent=2)
    else:
        # Default to CSV
        output_path += '.csv'
        df.to_csv(output_path, index=False)
    
    print(f"💾 Cleaned dataset saved to: {output_path}")

def main():
    """Main EDA and preprocessing tool"""
    parser = argparse.ArgumentParser(description='EDA and Preprocessing Tool for Medical Datasets')
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    parser.add_argument('--json', type=str, help='Path to JSON file')
    parser.add_argument('--output', type=str, help='Output path for cleaned dataset')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze, do not clean')
    
    args = parser.parse_args()
    
    if not args.csv and not args.json:
        print("❌ Please provide at least one dataset (--csv or --json)")
        return 1
    
    try:
        # Load dataset
        print("🚀 Medical Dataset EDA and Preprocessing Tool")
        print("=" * 60)
        
        df, dataset_name = load_dataset(args.csv, args.json)
        
        if args.analyze_only:
            print("\n🔍 Analysis Mode - No cleaning will be performed")
            # Just show basic info without cleaning
            print(f"📊 Dataset: {dataset_name}")
            print(f"📊 Shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
            print(f"📊 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Basic statistics
            print(f"\n📈 Basic Statistics:")
            print(df.describe(include='all'))
            
            # Check for missing values
            print(f"\n❌ Missing Values:")
            missing = df.isnull().sum()
            for col, count in missing.items():
                if count > 0:
                    print(f"   {col}: {count:,} ({count/len(df)*100:.1f}%)")
            
            return 0
        
        # Full EDA and preprocessing
        df_cleaned = perform_eda_and_preprocessing(df, dataset_name)
        
        if df_cleaned is None or len(df_cleaned) == 0:
            print("❌ Preprocessing resulted in empty dataset")
            return 1
        
        # Save cleaned dataset if output path provided
        if args.output:
            save_cleaned_dataset(df_cleaned, args.output)
        
        print(f"\n🎉 EDA and preprocessing completed successfully!")
        print(f"📊 Original: {len(df):,} records")
        print(f"📊 Cleaned: {len(df_cleaned):,} records")
        print(f"📊 Removed: {len(df) - len(df_cleaned):,} records")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())