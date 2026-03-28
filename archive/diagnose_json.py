#!/usr/bin/env python3
"""
Diagnostic script to analyze JSON file structure and content
"""

import json
import pandas as pd
import os

def diagnose_json_file(json_path):
    """Analyze JSON file structure and content"""
    print(f"🔍 Diagnosing JSON file: {json_path}")
    print("=" * 60)
    
    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        return
    
    try:
        # Load JSON data
        print("🔄 Loading JSON file...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Successfully loaded JSON file")
        print(f"📊 Data type: {type(data)}")
        
        if isinstance(data, list):
            print(f"📊 Number of records: {len(data):,}")
            
            if len(data) > 0:
                print(f"📋 First record keys: {list(data[0].keys())}")
                print(f"📋 First record sample:")
                for key, value in list(data[0].items())[:5]:  # Show first 5 fields
                    print(f"   {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
                
                # Convert to DataFrame for analysis
                df = pd.DataFrame(data)
                print(f"\n📊 DataFrame shape: {df.shape}")
                print(f"📋 DataFrame columns: {list(df.columns)}")
                print(f"📋 Column data types:")
                for col in df.columns:
                    print(f"   {col}: {df[col].dtype}")
                
                # Check for potential symptoms and disease columns
                print(f"\n🔍 Looking for symptoms and disease columns...")
                potential_symptoms = []
                potential_diseases = []
                
                for col in df.columns:
                    col_lower = col.lower().strip()
                    if any(x in col_lower for x in ['symptom', 'complaint', 'description', 'text']):
                        potential_symptoms.append(col)
                    if any(x in col_lower for x in ['disease', 'diagnosis', 'condition', 'label', 'target']):
                        potential_diseases.append(col)
                
                print(f"🎯 Potential symptoms columns: {potential_symptoms}")
                print(f"🎯 Potential disease columns: {potential_diseases}")
                
                # Sample data from potential columns
                if potential_symptoms:
                    print(f"\n📝 Sample from '{potential_symptoms[0]}':")
                    sample_values = df[potential_symptoms[0]].dropna().head(3)
                    for i, val in enumerate(sample_values):
                        print(f"   {i+1}: {str(val)[:150]}{'...' if len(str(val)) > 150 else ''}")
                
                if potential_diseases:
                    print(f"\n📝 Sample from '{potential_diseases[0]}':")
                    sample_values = df[potential_diseases[0]].dropna().head(3)
                    for i, val in enumerate(sample_values):
                        print(f"   {i+1}: {str(val)[:150]}{'...' if len(str(val)) > 150 else ''}")
                
                # Check for missing values
                print(f"\n📊 Missing values per column:")
                missing = df.isnull().sum()
                for col in df.columns:
                    if missing[col] > 0:
                        print(f"   {col}: {missing[col]:,} missing ({missing[col]/len(df)*100:.1f}%)")
                
                # Check unique values in potential disease column
                if potential_diseases:
                    disease_col = potential_diseases[0]
                    unique_diseases = df[disease_col].value_counts()
                    print(f"\n🎯 Unique diseases in '{disease_col}': {len(unique_diseases)}")
                    print(f"📊 Top 10 diseases:")
                    for disease, count in unique_diseases.head(10).items():
                        print(f"   {disease}: {count:,} records")
                
            else:
                print("⚠️  JSON file is empty")
        else:
            print(f"⚠️  Expected list format, got {type(data)}")
            print(f"📊 Keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
    
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print("💡 The file might not be valid JSON format")
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")

def main():
    """Main diagnostic function"""
    json_path = r"C:\Users\ayush\OneDrive\Desktop\diagnosis_data.json"
    
    print("🚀 JSON File Diagnostic Tool")
    print("=" * 60)
    
    diagnose_json_file(json_path)
    
    print("\n" + "=" * 60)
    print("💡 Recommendations:")
    print("1. Check if your JSON has 'symptoms' and 'disease' columns")
    print("2. If not, the robust loader will try to find alternatives")
    print("3. The model will work with any text columns for symptoms")
    print("4. Disease column can be any categorical column")

if __name__ == "__main__":
    main()