# 🧹 EDA and Preprocessing Pipeline

This document explains the comprehensive EDA (Exploratory Data Analysis) and preprocessing pipeline that has been added to your ML model.

## 🎯 **Overview**

The preprocessing pipeline automatically analyzes any dataset you feed to the model and performs comprehensive cleaning and validation before training begins.

## 🔍 **What the Pipeline Does**

### **1. Data Loading & Validation**
- Loads CSV and/or JSON datasets
- Validates required columns (symptoms, disease)
- Automatically maps alternative column names
- Handles missing or malformed data gracefully

### **2. Comprehensive EDA Analysis**
- **Dataset Statistics**: Size, shape, memory usage
- **Data Types**: Column types and missing values
- **Duplicate Analysis**: Identifies and counts duplicates
- **Class Distribution**: Disease frequency and imbalance
- **Symptoms Analysis**: Length statistics and quality checks
- **Data Quality**: Identifies empty, invalid, or problematic entries

### **3. Automatic Data Cleaning**
- **Removes Duplicates**: Eliminates exact duplicates
- **Cleans Invalid Data**: Removes empty, too short, or malformed entries
- **Standardizes Format**: Ensures consistent data types
- **Handles Missing Values**: Properly manages null values

### **4. Cross-Validation Readiness Check**
- Validates minimum class requirements
- Checks for sufficient samples per class
- Ensures data is ready for ML training

## 🚀 **Usage**

### **Automatic (Built into Model)**
The preprocessing runs automatically when you train the model:

```bash
python models/sihdemo.py --csv "your_data.csv" --json "your_data.json"
```

### **Standalone Analysis Tool**
Use the standalone tool to analyze datasets without training:

```bash
# Analyze only (no cleaning)
python eda_preprocessing_tool.py --csv "data.csv" --analyze-only

# Full EDA and preprocessing
python eda_preprocessing_tool.py --csv "data.csv" --json "data.json" --output "cleaned_data.csv"
```

## 📊 **Example Output**

```
🔍 EDA and Preprocessing for Combined Dataset
============================================================
📊 Original dataset size: 100,000 records
📋 Columns: ['symptoms', 'disease', 'age', 'gender', 'severity']
📊 Shape: (100000, 5)

📈 Data Types:
   symptoms: object (missing: 0)
   disease: object (missing: 0)
   age: int64 (missing: 1,234)
   gender: object (missing: 567)
   severity: float64 (missing: 0)

🧹 Cleaning symptoms data...
   Removed 1,500 rows with missing symptoms/disease
   Removed 2,300 rows with invalid symptoms/disease

🔍 Duplicate Analysis:
   Total duplicates: 15,000
   Symptom duplicates: 8,500
   Full duplicates: 12,000
   Removed 12,000 duplicate records

🎯 Disease Distribution Analysis:
   Total unique diseases: 13
   Most common diseases:
     Healthy: 12,000 (15.2%)
     Hypertension: 10,000 (12.7%)
     Dengue: 6,356 (8.1%)
     COVID-19: 6,351 (8.1%)
     Seasonal_Flu: 6,000 (7.6%)
   Class imbalance ratio: 12.5:1
   ⚠️  High class imbalance detected

📝 Symptoms Analysis:
   Average symptom length: 45.2 characters
   Min length: 4
   Max length: 234
   Median length: 42.0
   ⚠️  15 symptoms longer than 500 characters

🔍 Data Quality Checks:
   Empty symptoms: 0
   Numeric-only symptoms: 0
   Single character symptoms: 0

✅ Cross-Validation Readiness:
   ✅ Balanced classes

📊 Preprocessing Summary:
   Original records: 100,000
   Final records: 84,200
   Removed: 15,800 (15.8%)
   Data quality: ✅ Good
```

## 🛠️ **Configuration Options**

### **Column Mapping**
The pipeline automatically detects these column variations:

**Symptoms Columns:**
- `symptoms`, `symptom`, `complaint`, `complaints`
- `description`, `desc`, `text`

**Disease Columns:**
- `disease`, `diagnosis`, `condition`
- `illness`, `disorder`, `label`, `target`, `outcome`

### **Data Quality Thresholds**
- **Minimum symptom length**: 3 characters
- **Minimum disease length**: 1 character
- **Very long symptoms**: >500 characters (flagged)
- **Class imbalance warning**: >5:1 ratio
- **High class imbalance**: >10:1 ratio

## 🎯 **Benefits**

### **1. Automatic Data Quality Assurance**
- No more manual data cleaning
- Consistent preprocessing across all datasets
- Automatic detection of data issues

### **2. Comprehensive Analysis**
- Detailed insights into your data
- Identifies potential problems before training
- Helps understand data characteristics

### **3. Robust Error Handling**
- Handles missing columns gracefully
- Manages different data formats
- Provides clear error messages

### **4. Training Optimization**
- Removes problematic data that could hurt training
- Ensures data is ready for cross-validation
- Prevents data leakage issues

## ⚠️ **Important Notes**

1. **Data Loss**: The pipeline may remove some records that don't meet quality standards
2. **Backup**: Always keep a backup of your original data
3. **Review**: Check the preprocessing summary to understand what was removed
4. **Customization**: You can modify thresholds in the `perform_eda_and_preprocessing()` function

## 🔧 **Troubleshooting**

### **High Data Removal Rate**
If >50% of data is removed:
- Check for systematic data quality issues
- Review column mappings
- Verify data format

### **Class Imbalance Warnings**
- Consider data augmentation
- Use stratified sampling
- Apply class weighting

### **Missing Required Columns**
- Check column names in your dataset
- Use the standalone tool to analyze first
- Verify data format (CSV/JSON)

## 🎉 **Result**

After preprocessing, your model will train on clean, high-quality data that is:
- ✅ **Duplicate-free**
- ✅ **Properly formatted**
- ✅ **Cross-validation ready**
- ✅ **Data leakage free**
- ✅ **Quality assured**

This ensures reliable, realistic model performance! 🚀