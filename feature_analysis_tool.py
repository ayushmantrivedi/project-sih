#!/usr/bin/env python3
"""
Comprehensive Feature Analysis Tool for Medical Datasets
Analyzes features and provides recommendations for model improvement
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import argparse

def analyze_feature_importance(df, target_col='disease'):
    """Analyze feature importance using multiple methods"""
    print("🔬 Feature Importance Analysis")
    print("=" * 50)
    
    # Prepare data
    X = df.drop(columns=[target_col, 'symptoms'] if 'symptoms' in df.columns else [target_col])
    y = df[target_col]
    
    # Encode categorical variables
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Encode categorical features
    X_encoded = X.copy()
    for col in X_encoded.columns:
        if X_encoded[col].dtype == 'object':
            X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))
    
    # Fill missing values
    X_encoded = X_encoded.fillna(X_encoded.median())
    
    print(f"📊 Analyzing {len(X.columns)} features against {len(le.classes_)} disease classes")
    
    # 1. Mutual Information
    print("\n📈 Mutual Information with Disease:")
    try:
        mi_scores = mutual_info_classif(X_encoded, y_encoded, random_state=42)
        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mutual_info': mi_scores
        }).sort_values('mutual_info', ascending=False)
        
        for _, row in mi_df.head(10).iterrows():
            print(f"   {row['feature']}: {row['mutual_info']:.4f}")
    except Exception as e:
        print(f"   Error calculating mutual information: {e}")
    
    # 2. Random Forest Feature Importance
    print("\n🌲 Random Forest Feature Importance:")
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_encoded, y_encoded)
        
        rf_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for _, row in rf_importance.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
    except Exception as e:
        print(f"   Error calculating RF importance: {e}")
    
    # 3. Correlation Analysis
    print("\n📊 Correlation with Disease (encoded):")
    try:
        correlations = []
        for col in X_encoded.columns:
            if X_encoded[col].dtype in ['int64', 'float64']:
                corr = X_encoded[col].corr(pd.Series(y_encoded))
                correlations.append((col, abs(corr)))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        for col, corr in correlations[:10]:
            print(f"   {col}: {corr:.4f}")
    except Exception as e:
        print(f"   Error calculating correlations: {e}")
    
    return mi_df, rf_importance

def analyze_class_distribution(df, target_col='disease'):
    """Analyze class distribution and imbalance"""
    print("\n🎯 Class Distribution Analysis")
    print("=" * 50)
    
    class_counts = df[target_col].value_counts()
    total_samples = len(df)
    
    print(f"📊 Total samples: {total_samples:,}")
    print(f"📊 Number of classes: {len(class_counts)}")
    print(f"📊 Class distribution:")
    
    for disease, count in class_counts.items():
        percentage = count / total_samples * 100
        print(f"   {disease}: {count:,} ({percentage:.1f}%)")
    
    # Calculate imbalance metrics
    min_class = class_counts.min()
    max_class = class_counts.max()
    imbalance_ratio = max_class / min_class
    
    print(f"\n📈 Imbalance Analysis:")
    print(f"   Smallest class: {min_class:,} samples")
    print(f"   Largest class: {max_class:,} samples")
    print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > 10:
        print("   ⚠️  Severe class imbalance - consider resampling or class weights")
    elif imbalance_ratio > 5:
        print("   ⚠️  Moderate class imbalance - consider class weights")
    else:
        print("   ✅ Balanced classes")
    
    return class_counts

def analyze_symptom_patterns(df, symptoms_col='symptoms'):
    """Analyze symptom patterns and quality"""
    print("\n📝 Symptom Pattern Analysis")
    print("=" * 50)
    
    symptoms = df[symptoms_col].astype(str)
    
    # Basic statistics
    lengths = symptoms.str.len()
    print(f"📊 Symptom Statistics:")
    print(f"   Average length: {lengths.mean():.1f} characters")
    print(f"   Median length: {lengths.median():.1f} characters")
    print(f"   Min length: {lengths.min()}")
    print(f"   Max length: {lengths.max()}")
    print(f"   Std deviation: {lengths.std():.1f}")
    
    # Length distribution
    print(f"\n📈 Length Distribution:")
    length_bins = [0, 10, 20, 30, 50, 100, float('inf')]
    length_labels = ['0-10', '11-20', '21-30', '31-50', '51-100', '100+']
    
    for i in range(len(length_bins)-1):
        count = ((lengths >= length_bins[i]) & (lengths < length_bins[i+1])).sum()
        percentage = count / len(symptoms) * 100
        print(f"   {length_labels[i]} chars: {count:,} ({percentage:.1f}%)")
    
    # Common symptom words
    print(f"\n🔤 Most Common Symptom Words:")
    all_symptoms = ' '.join(symptoms.str.lower())
    words = all_symptoms.split()
    word_counts = pd.Series(words).value_counts()
    
    for word, count in word_counts.head(15).items():
        if len(word) > 2:  # Skip very short words
            print(f"   '{word}': {count:,} times")
    
    # Quality issues
    print(f"\n🔍 Quality Issues:")
    empty_symptoms = (symptoms.str.strip() == '').sum()
    very_short = (lengths < 5).sum()
    very_long = (lengths > 200).sum()
    numeric_only = symptoms.str.isnumeric().sum()
    
    print(f"   Empty symptoms: {empty_symptoms}")
    print(f"   Very short (<5 chars): {very_short}")
    print(f"   Very long (>200 chars): {very_long}")
    print(f"   Numeric only: {numeric_only}")
    
    return {
        'avg_length': lengths.mean(),
        'min_length': lengths.min(),
        'max_length': lengths.max(),
        'quality_score': 1 - (empty_symptoms + very_short + numeric_only) / len(symptoms)
    }

def generate_recommendations(df, target_col='disease', symptoms_col='symptoms'):
    """Generate recommendations for model improvement"""
    print("\n💡 Model Improvement Recommendations")
    print("=" * 50)
    
    # Analyze current state
    class_counts = df[target_col].value_counts()
    symptoms_stats = analyze_symptom_patterns(df, symptoms_col)
    
    recommendations = []
    
    # Class imbalance recommendations
    imbalance_ratio = class_counts.max() / class_counts.min()
    if imbalance_ratio > 10:
        recommendations.append({
            'issue': 'Severe class imbalance',
            'impact': 'High',
            'solution': 'Use class weights, SMOTE, or stratified sampling',
            'code': 'from sklearn.utils.class_weight import compute_class_weight'
        })
    elif imbalance_ratio > 5:
        recommendations.append({
            'issue': 'Moderate class imbalance',
            'impact': 'Medium',
            'solution': 'Use class weights or balanced sampling',
            'code': 'class_weight="balanced"'
        })
    
    # Symptom quality recommendations
    if symptoms_stats['avg_length'] < 20:
        recommendations.append({
            'issue': 'Short symptom descriptions',
            'impact': 'High',
            'solution': 'Enhance symptom descriptions with more details',
            'code': 'Add severity, duration, and associated symptoms'
        })
    
    if symptoms_stats['quality_score'] < 0.9:
        recommendations.append({
            'issue': 'Low symptom quality',
            'impact': 'Medium',
            'solution': 'Clean and standardize symptom descriptions',
            'code': 'Remove empty, very short, or invalid entries'
        })
    
    # Feature recommendations
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    categorical_features = [col for col in categorical_features if col not in [target_col, symptoms_col]]
    
    if len(numeric_features) < 3:
        recommendations.append({
            'issue': 'Limited numeric features',
            'impact': 'Medium',
            'solution': 'Add more clinical features (age, vital signs, lab values)',
            'code': 'Include age, severity, duration, temperature, etc.'
        })
    
    if len(categorical_features) < 2:
        recommendations.append({
            'issue': 'Limited categorical features',
            'impact': 'Low',
            'solution': 'Add demographic and clinical categorical features',
            'code': 'Include gender, medical history, family history, etc.'
        })
    
    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['issue']}")
        print(f"   Impact: {rec['impact']}")
        print(f"   Solution: {rec['solution']}")
        print(f"   Implementation: {rec['code']}")
    
    if not recommendations:
        print("✅ No major issues detected. Your dataset looks good for training!")
    
    return recommendations

def main():
    """Main feature analysis function"""
    parser = argparse.ArgumentParser(description='Comprehensive Feature Analysis Tool')
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    parser.add_argument('--json', type=str, help='Path to JSON file')
    parser.add_argument('--target', type=str, default='disease', help='Target column name')
    parser.add_argument('--symptoms', type=str, default='symptoms', help='Symptoms column name')
    
    args = parser.parse_args()
    
    if not args.csv and not args.json:
        print("❌ Please provide at least one dataset (--csv or --json)")
        return 1
    
    try:
        # Load dataset
        print("🚀 Comprehensive Feature Analysis Tool")
        print("=" * 60)
        
        datasets = []
        if args.csv and os.path.exists(args.csv):
            df_csv = pd.read_csv(args.csv)
            datasets.append(("CSV", df_csv))
        
        if args.json and os.path.exists(args.json):
            with open(args.json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df_json = pd.DataFrame(data)
            datasets.append(("JSON", df_json))
        
        if len(datasets) == 1:
            name, df = datasets[0]
        else:
            df = pd.concat([df for _, df in datasets], ignore_index=True)
            name = "Combined"
        
        print(f"📊 Analyzing {name} dataset: {len(df):,} records")
        
        # Run analyses
        analyze_class_distribution(df, args.target)
        analyze_symptom_patterns(df, args.symptoms)
        analyze_feature_importance(df, args.target)
        generate_recommendations(df, args.target, args.symptoms)
        
        print(f"\n🎉 Feature analysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())