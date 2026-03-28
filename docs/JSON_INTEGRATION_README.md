# JSON Data Integration with ML Model

This document explains how the ML model (`sihdemo.py`) has been enhanced to integrate JSON data alongside the existing CSV database for improved training and prediction accuracy.

## 🎯 Overview

The enhanced ML model now supports training on both:
- **CSV Data**: `C:\Users\ayush\OneDrive\Desktop\augmented_synthetic_health_dataset.csv`
- **JSON Data**: `C:\Users\ayush\OneDrive\Desktop\diagnosis_data.json`

This dual-data approach provides:
- ✅ More diverse training data
- ✅ Better model generalization
- ✅ Improved prediction accuracy
- ✅ Flexible data format support

## 📁 File Structure

```
/workspace/
├── models/
│   └── sihdemo.py                    # Enhanced ML model with JSON integration
├── diagnosis_data.json               # Sample JSON training data
├── test_json_integration.py          # Integration tests
├── train_with_json.py               # Training script
├── api_with_json_integration.py     # API integration example
└── JSON_INTEGRATION_README.md       # This documentation
```

## 🔧 Key Changes Made

### 1. Enhanced Data Loading (`sihdemo.py`)

- **Added JSON support**: New `load_json_data()` function
- **Combined datasets**: Modified `load_and_prepare()` to handle both CSV and JSON
- **Automatic merging**: Intelligently combines datasets with common columns
- **Error handling**: Robust error handling for missing or malformed data

### 2. Configuration Updates

```python
# New configuration
JSON_PATH = r"C:\Users\ayush\OneDrive\Desktop\diagnosis_data.json"
```

### 3. Command Line Arguments

```bash
python models/sihdemo.py --csv path/to/csv --json path/to/json --epochs 30
```

## 📊 JSON Data Format

The JSON file should contain an array of objects with the following structure:

```json
[
  {
    "symptoms": "fever headache body ache fatigue",
    "disease": "flu",
    "description": "Patient experiencing high temperature with severe headache",
    "age": 35,
    "gender": "male",
    "severity": 7
  }
]
```

### Required Fields:
- `symptoms`: String describing patient symptoms
- `disease`: String indicating the predicted disease

### Optional Fields:
- `description`: Additional symptom description
- `age`: Patient age (numeric)
- `gender`: Patient gender (string)
- `severity`: Symptom severity (numeric)

## 🚀 Usage Examples

### 1. Training with Both Datasets

```bash
# Train with both CSV and JSON data
python train_with_json.py --csv your_data.csv --json your_data.json --epochs 30

# Or use the model directly
python models/sihdemo.py --csv your_data.csv --json your_data.json
```

### 2. Testing Integration

```bash
# Run integration tests
python test_json_integration.py
```

### 3. API Usage

```bash
# Start the API server
python api_with_json_integration.py
```

#### API Endpoints:

**Health Check:**
```bash
curl http://localhost:5000/health
```

**Single Prediction:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "fever headache body ache", "numeric_features": [25, 1, 7]}'
```

**Batch Prediction:**
```bash
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms_list": [
      {"symptoms": "fever headache", "numeric_features": [25, 1, 7]},
      {"symptoms": "cough chest pain", "numeric_features": [45, 0, 8]}
    ]
  }'
```

## 🔍 How It Works

### 1. Data Loading Process

1. **Load CSV data** from the specified path
2. **Load JSON data** from the specified path (if exists)
3. **Find common columns** between both datasets
4. **Merge datasets** using common columns only
5. **Process combined data** through the existing pipeline

### 2. Training Process

1. **Text preprocessing** on symptoms and descriptions
2. **Data augmentation** using synonym replacement
3. **Feature extraction** using Bio_ClinicalBERT embeddings
4. **Numeric feature scaling** and normalization
5. **Model training** with global-local line search optimization
6. **Cross-validation** for robust evaluation

### 3. Prediction Process

1. **Text preprocessing** of input symptoms
2. **Feature extraction** using the trained encoder
3. **Numeric feature processing** (if provided)
4. **Feature fusion** using attention mechanism
5. **Disease prediction** with confidence scores

## 📈 Benefits

### 1. Improved Data Diversity
- More training examples from different sources
- Better coverage of symptom-disease combinations
- Reduced overfitting through data variety

### 2. Enhanced Accuracy
- Combined knowledge from multiple datasets
- Better generalization to unseen cases
- More robust feature representations

### 3. Flexible Data Management
- Support for different data formats
- Easy addition of new data sources
- Scalable architecture for future enhancements

## 🧪 Testing

The integration includes comprehensive tests:

```bash
python test_json_integration.py
```

**Test Coverage:**
- ✅ JSON data loading
- ✅ Combined CSV + JSON loading
- ✅ Data format validation
- ✅ Error handling
- ✅ Model prediction (if trained)

## 🔧 Customization

### Adding New Data Sources

1. **Create data loader function:**
```python
def load_your_data(data_path):
    # Your data loading logic
    return pd.DataFrame(your_data)
```

2. **Update `load_and_prepare()`:**
```python
# Add your data source
df_your = load_your_data(your_path)
# Merge with existing data
```

### Modifying Data Format

Update the JSON structure in `load_json_data()` to match your format:

```python
# Modify column mapping as needed
required_columns = ["symptoms", "disease"]
optional_columns = ["description", "age", "gender", "severity"]
```

## 🚨 Troubleshooting

### Common Issues:

1. **File not found errors:**
   - Ensure CSV and JSON files exist at specified paths
   - Check file permissions

2. **Column mismatch errors:**
   - Ensure both datasets have common columns
   - Check column names match exactly

3. **Memory issues:**
   - Reduce batch size for large datasets
   - Use data sampling for very large datasets

4. **Model loading errors:**
   - Train the model first before using API
   - Check model file paths

## 📝 Notes

- The JSON data is automatically validated for required columns
- Missing optional columns are handled gracefully
- Data augmentation is applied only to training data
- The model maintains backward compatibility with CSV-only training

## 🎉 Success Metrics

After integration, you should see:
- ✅ Successful loading of both CSV and JSON data
- ✅ Combined dataset with more training examples
- ✅ Improved model accuracy on validation data
- ✅ Better API prediction results
- ✅ All integration tests passing

The enhanced model now provides more accurate predictions by leveraging the combined knowledge from both your CSV database and the JSON data file!