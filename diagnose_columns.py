#!/usr/bin/env python3
"""
Diagnostic script to analyze column structure of CSV and JSON files
"""

import os
import sys
import pandas as pd
import json
import argparse

def analyze_csv_columns(csv_path):
    """Analyze CSV file columns"""
    print(f"📊 CSV File Analysis: {csv_path}")
    print("=" * 50)
    
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Successfully loaded CSV")
        print(f"📊 Shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"📊 Data types:")
        for col in df.columns:
            print(f"   {col}: {df[col].dtype}")
        
        print(f"\n📝 Sample data (first 3 rows):")
        print(df.head(3).to_string())
        
        return df.columns.tolist()
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None

def analyze_json_columns(json_path):
    """Analyze JSON file columns"""
    print(f"\n📊 JSON File Analysis: {json_path}")
    print("=" * 50)
    
    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            print("❌ JSON file is empty")
            return None
        
        df = pd.DataFrame(data)
        print(f"✅ Successfully loaded JSON")
        print(f"📊 Shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"📊 Data types:")
        for col in df.columns:
            print(f"   {col}: {df[col].dtype}")
        
        print(f"\n📝 Sample data (first 3 rows):")
        print(df.head(3).to_string())
        
        return df.columns.tolist()
        
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return None

def compare_columns(csv_cols, json_cols):
    """Compare columns between CSV and JSON"""
    print(f"\n🔍 Column Comparison Analysis")
    print("=" * 50)
    
    if csv_cols is None or json_cols is None:
        print("❌ Cannot compare - one or both files failed to load")
        return
    
    csv_set = set(csv_cols)
    json_set = set(json_cols)
    
    common_cols = csv_set & json_set
    csv_only = csv_set - json_set
    json_only = json_set - csv_set
    
    print(f"🔗 Common columns ({len(common_cols)}): {list(common_cols)}")
    print(f"📊 CSV-only columns ({len(csv_only)}): {list(csv_only)}")
    print(f"📊 JSON-only columns ({len(json_only)}): {list(json_only)}")
    
    # Check for required columns
    required = {'symptoms', 'disease'}
    has_required = required.issubset(common_cols)
    
    print(f"\n✅ Required columns check:")
    print(f"   Symptoms: {'✅' if 'symptoms' in common_cols else '❌'}")
    print(f"   Disease: {'✅' if 'disease' in common_cols else '❌'}")
    print(f"   Both required: {'✅' if has_required else '❌'}")
    
    # Feature analysis
    print(f"\n🎯 Feature Analysis:")
    all_cols = csv_set | json_set
    feature_cols = all_cols - required
    
    print(f"   Total unique columns: {len(all_cols)}")
    print(f"   Feature columns (excluding symptoms/disease): {len(feature_cols)}")
    print(f"   Feature columns: {list(feature_cols)}")
    
    if len(feature_cols) > 0:
        print(f"   ✅ Good feature diversity for multi-modal model")
    else:
        print(f"   ⚠️  Limited features - model will rely mainly on symptoms")

def main():
    """Main diagnostic function"""
    parser = argparse.ArgumentParser(description='Diagnose CSV and JSON column structure')
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    parser.add_argument('--json', type=str, help='Path to JSON file')
    
    args = parser.parse_args()
    
    if not args.csv and not args.json:
        print("❌ Please provide at least one file (--csv or --json)")
        return 1
    
    print("🔍 Column Structure Diagnostic Tool")
    print("=" * 60)
    
    csv_cols = None
    json_cols = None
    
    if args.csv:
        csv_cols = analyze_csv_columns(args.csv)
    
    if args.json:
        json_cols = analyze_json_columns(args.json)
    
    if csv_cols is not None and json_cols is not None:
        compare_columns(csv_cols, json_cols)
    
    print(f"\n🎉 Column analysis completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())