#!/usr/bin/env python3
"""
GPU-Optimized Training Script for HealthBot AI Chatbot
Features:
- GPU acceleration with CUDA
- Mixed precision training
- Data parallelism for multi-GPU
- Memory optimization
- Parallel data loading
- Optimized batch processing
"""

import os
import sys
import gc
import time
import argparse
import psutil
from typing import List, Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# Set environment variables for GPU optimization
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertModel, AutoTokenizer, AutoModel
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import re

# Configuration
CONFIG = {
    "MODEL_NAME": "emilyalsentzer/Bio_ClinicalBERT",
    "MAX_LEN": 128,  # Increased for better context
    "BATCH_SIZE": 64,  # Increased for GPU
    "NUM_EPOCHS": 25,
    "BASE_LR": 3e-5,
    "SEED": 42,
    "MIXED_PRECISION": True,
    "USE_MULTI_GPU": False,  # Set to True if you have multiple GPUs
    "NUM_WORKERS": min(8, mp.cpu_count()),
    "PREFETCH_BUFFER": 4,
    "CACHE_DATASET": True,
    "EARLY_STOPPING_PATIENCE": 7,
    "LEARNING_RATE_SCHEDULE": True,
    "WEIGHT_DECAY": 1e-4,
    "GRADIENT_CLIPPING": 1.0,
    "WARMUP_STEPS": 1000,
}

def setup_gpu():
    """Setup GPU configuration and check availability"""
    print("🔧 Setting up GPU configuration...")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("⚠️  No GPU detected. Training will use CPU (slower)")
        return False
    
    print(f"✅ Found {len(gpus)} GPU(s)")
    
    # Configure GPU memory growth
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled")
    except RuntimeError as e:
        print(f"⚠️  GPU memory growth setup failed: {e}")
    
    # Enable mixed precision if supported
    if CONFIG["MIXED_PRECISION"]:
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✅ Mixed precision training enabled")
        except Exception as e:
            print(f"⚠️  Mixed precision setup failed: {e}")
            CONFIG["MIXED_PRECISION"] = False
    
    # Setup multi-GPU if available and requested
    if CONFIG["USE_MULTI_GPU"] and len(gpus) > 1:
        try:
            strategy = tf.distribute.MirroredStrategy()
            print(f"✅ Multi-GPU strategy enabled with {strategy.num_replicas_in_sync} GPUs")
            return strategy
        except Exception as e:
            print(f"⚠️  Multi-GPU setup failed: {e}")
            CONFIG["USE_MULTI_GPU"] = False
    
    return True

def set_seed(seed: int = CONFIG["SEED"]):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def normalize_text(text: str) -> str:
    """Enhanced text normalization"""
    if not isinstance(text, str):
        text = str(text)
    
    text = text.lower().strip()
    
    # Remove punctuation and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove medical stopwords
    medical_stopwords = {
        "mg", "ml", "tablet", "dose", "day", "week", "month", "year",
        "patient", "history", "present", "complaint", "chief"
    }
    
    tokens = text.split()
    tokens = [t for t in tokens if t not in medical_stopwords and len(t) > 2]
    
    return " ".join(tokens)

def create_optimized_dataset(csv_path: str, json_path: Optional[str] = None, 
                           sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Create optimized dataset with parallel processing"""
    print("📊 Creating optimized dataset...")
    
    def load_csv_parallel(csv_path: str) -> pd.DataFrame:
        """Load CSV with parallel processing"""
        try:
            # Use chunking for large files
            chunk_size = 10000
            chunks = []
            
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                chunks.append(chunk)
                if len(chunks) * chunk_size > 100000:  # Limit memory usage
                    break
            
            df = pd.concat(chunks, ignore_index=True)
            print(f"✅ Loaded {len(df):,} records from CSV")
            return df
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return pd.DataFrame()
    
    def load_json_parallel(json_path: str) -> pd.DataFrame:
        """Load JSON with parallel processing"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            print(f"✅ Loaded {len(df):,} records from JSON")
            return df
        except Exception as e:
            print(f"❌ Error loading JSON: {e}")
            return pd.DataFrame()
    
    # Load data in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        csv_future = executor.submit(load_csv_parallel, csv_path)
        json_future = executor.submit(load_json_parallel, json_path) if json_path else None
        
        df_csv = csv_future.result()
        df_json = json_future.result() if json_future else pd.DataFrame()
    
    # Combine datasets
    if not df_json.empty:
        # Find common columns
        common_cols = set(df_csv.columns) & set(df_json.columns)
        if common_cols:
            df_csv = df_csv[list(common_cols)]
            df_json = df_json[list(common_cols)]
            df = pd.concat([df_csv, df_json], ignore_index=True)
        else:
            df = df_csv
    else:
        df = df_csv
    
    # Apply sampling if requested
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=CONFIG["SEED"])
        print(f"📊 Sampled to {len(df):,} records")
    
    # Clean and prepare data
    print("🧹 Cleaning and preparing data...")
    
    # Find text and label columns
    text_col = None
    label_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if any(x in col_lower for x in ['symptom', 'complaint', 'description', 'text']):
            text_col = col
        if any(x in col_lower for x in ['disease', 'diagnosis', 'condition', 'label']):
            label_col = col
    
    if not text_col or not label_col:
        raise ValueError("Could not find suitable text and label columns")
    
    # Clean data
    df = df.dropna(subset=[text_col, label_col])
    df[text_col] = df[text_col].apply(normalize_text)
    df = df[df[text_col].str.len() > 3]
    
    # Identify numeric features
    exclude = {text_col, label_col}
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    
    # Fill missing numeric values
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(0.0)
    
    print(f"✅ Dataset prepared: {len(df):,} records, {len(numeric_cols)} numeric features")
    
    return df, numeric_cols

def create_optimized_model(input_dim: int, num_classes: int, 
                          text_dim: int = 768, numeric_dim: int = 0,
                          strategy=None) -> tf.keras.Model:
    """Create optimized model architecture for GPU training"""
    
    def build_model():
        inputs = tf.keras.Input(shape=(input_dim,), dtype=tf.float32, name="input")
        
        # Separate text and numeric features
        text_features = tf.keras.layers.Lambda(lambda x: x[:, :text_dim])(inputs)
        numeric_features = tf.keras.layers.Lambda(lambda x: x[:, text_dim:])(inputs) if numeric_dim > 0 else None
        
        # Enhanced text processing branch
        text_branch = tf.keras.layers.Dense(1024, activation="gelu", name="text_dense1")(text_features)
        text_branch = tf.keras.layers.BatchNormalization()(text_branch)
        text_branch = tf.keras.layers.Dropout(0.3)(text_branch)
        
        text_branch = tf.keras.layers.Dense(512, activation="gelu", name="text_dense2")(text_branch)
        text_branch = tf.keras.layers.BatchNormalization()(text_branch)
        text_branch = tf.keras.layers.Dropout(0.3)(text_branch)
        
        # Numeric processing branch
        if numeric_dim > 0:
            numeric_branch = tf.keras.layers.Dense(max(128, numeric_dim * 4), activation="gelu")(numeric_features)
            numeric_branch = tf.keras.layers.BatchNormalization()(numeric_branch)
            numeric_branch = tf.keras.layers.Dropout(0.2)(numeric_branch)
            
            # Feature fusion with attention
            fused = tf.keras.layers.Concatenate()([text_branch, numeric_branch])
            attention = tf.keras.layers.Dense(1, activation="sigmoid")(fused)
            attended = tf.keras.layers.Multiply()([fused, attention])
        else:
            attended = text_branch
        
        # Final classification layers
        x = tf.keras.layers.Dense(256, activation="gelu")(attended)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        x = tf.keras.layers.Dense(128, activation="gelu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(num_classes, name="outputs")(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="gpu_optimized_classifier")
        return model
    
    if strategy:
        with strategy.scope():
            return build_model()
    else:
        return build_model()

def create_optimized_optimizer(num_train_steps: int) -> tf.keras.optimizers.Optimizer:
    """Create optimized optimizer with learning rate scheduling"""
    
    # Learning rate schedule
    if CONFIG["LEARNING_RATE_SCHEDULE"]:
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=CONFIG["BASE_LR"],
            decay_steps=num_train_steps,
            end_learning_rate=CONFIG["BASE_LR"] * 0.1,
            power=0.5
        )
    else:
        lr_schedule = CONFIG["BASE_LR"]
    
    # Create optimizer
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=CONFIG["WEIGHT_DECAY"],
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )
    
    return optimizer

def create_optimized_dataset_tf(texts: List[str], labels: List[int], 
                               numeric_features: np.ndarray,
                               tokenizer, encoder, 
                               batch_size: int = CONFIG["BATCH_SIZE"],
                               cache: bool = CONFIG["CACHE_DATASET"]) -> tf.data.Dataset:
    """Create optimized TensorFlow dataset with parallel processing"""
    
    def process_text(text):
        """Process single text"""
        processed = normalize_text(text)
        return processed
    
    def encode_text(text):
        """Encode text to embeddings"""
        encoding = tokenizer(
            text.numpy().decode('utf-8'),
            truncation=True,
            padding='max_length',
            max_length=CONFIG["MAX_LEN"],
            return_tensors='tf'
        )
        
        # Get embeddings
        outputs = encoder(encoding, training=False)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        return embeddings
    
    def tf_encode_text(text):
        """TensorFlow wrapper for text encoding"""
        return tf.py_function(encode_text, [text], tf.float32)
    
    # Process texts in parallel
    print("🔄 Processing texts in parallel...")
    with ThreadPoolExecutor(max_workers=CONFIG["NUM_WORKERS"]) as executor:
        processed_texts = list(executor.map(process_text, texts))
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices({
        'text': processed_texts,
        'label': labels,
        'numeric': numeric_features
    })
    
    # Encode texts
    dataset = dataset.map(lambda x: {
        'text_emb': tf_encode_text(x['text']),
        'label': x['label'],
        'numeric': x['numeric']
    }, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Fuse features
    def fuse_features(x):
        if x['numeric'].shape[0] > 0:
            fused = tf.concat([x['text_emb'], x['numeric']], axis=0)
        else:
            fused = x['text_emb']
        return fused, x['label']
    
    dataset = dataset.map(fuse_features, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Optimize dataset
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(CONFIG["PREFETCH_BUFFER"])
    
    if cache:
        dataset = dataset.cache()
    
    return dataset

def train_optimized_model(model: tf.keras.Model, 
                         train_dataset: tf.data.Dataset,
                         val_dataset: tf.data.Dataset,
                         num_classes: int,
                         strategy=None) -> tf.keras.Model:
    """Train model with GPU optimization"""
    
    print("🚀 Starting optimized training...")
    
    # Calculate training steps
    num_train_steps = sum(1 for _ in train_dataset) * CONFIG["NUM_EPOCHS"]
    num_warmup_steps = CONFIG["WARMUP_STEPS"]
    
    # Create optimizer
    optimizer = create_optimized_optimizer(num_train_steps)
    
    # Loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Metrics
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[train_accuracy]
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=CONFIG["EARLY_STOPPING_PATIENCE"],
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_gpu_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Training
    print(f"📊 Training for {CONFIG['NUM_EPOCHS']} epochs...")
    print(f"📊 Batch size: {CONFIG['BATCH_SIZE']}")
    print(f"📊 Learning rate: {CONFIG['BASE_LR']}")
    
    start_time = time.time()
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=CONFIG["NUM_EPOCHS"],
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"⏱️  Training completed in {training_time:.2f} seconds")
    
    return model, history

def evaluate_model(model: tf.keras.Model, test_dataset: tf.data.Dataset) -> Dict[str, float]:
    """Evaluate model performance"""
    print("📊 Evaluating model...")
    
    # Get predictions
    y_true = []
    y_pred = []
    
    for batch in test_dataset:
        features, labels = batch
        predictions = model(features, training=False)
        pred_labels = tf.argmax(predictions, axis=1)
        
        y_true.extend(labels.numpy())
        y_pred.extend(pred_labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"✅ Test Accuracy: {accuracy:.4f}")
    print(f"✅ Test F1-Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1
    }

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='GPU-Optimized ML Training')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV data')
    parser.add_argument('--json', type=str, default=None, help='Path to JSON data')
    parser.add_argument('--epochs', type=int, default=CONFIG["NUM_EPOCHS"])
    parser.add_argument('--batch_size', type=int, default=CONFIG["BATCH_SIZE"])
    parser.add_argument('--lr', type=float, default=CONFIG["BASE_LR"])
    parser.add_argument('--sample_size', type=int, default=None, help='Sample size for testing')
    parser.add_argument('--output_dir', type=str, default='gpu_optimized_model')
    
    args = parser.parse_args()
    
    # Update config
    CONFIG["NUM_EPOCHS"] = args.epochs
    CONFIG["BATCH_SIZE"] = args.batch_size
    CONFIG["BASE_LR"] = args.lr
    
    print("🚀 GPU-Optimized ML Training")
    print("=" * 50)
    print(f"📊 Epochs: {CONFIG['NUM_EPOCHS']}")
    print(f"📊 Batch size: {CONFIG['BATCH_SIZE']}")
    print(f"📊 Learning rate: {CONFIG['BASE_LR']}")
    print(f"📊 Mixed precision: {CONFIG['MIXED_PRECISION']}")
    print()
    
    # Setup
    set_seed()
    strategy = setup_gpu()
    
    # Load data
    df, numeric_cols = create_optimized_dataset(args.csv, args.json, args.sample_size)
    
    # Prepare features
    texts = df['symptoms'].tolist()
    labels = df['disease'].tolist()
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    
    # Prepare numeric features
    if numeric_cols:
        scaler = StandardScaler()
        X_numeric = scaler.fit_transform(df[numeric_cols].values)
    else:
        X_numeric = np.zeros((len(df), 0))
    
    # Split data
    X_train, X_test, y_train, y_test, X_num_train, X_num_test = train_test_split(
        texts, y_encoded, X_numeric, test_size=0.2, random_state=CONFIG["SEED"], stratify=y_encoded
    )
    
    X_train, X_val, y_train, y_val, X_num_train, X_num_val = train_test_split(
        X_train, y_train, X_num_train, test_size=0.2, random_state=CONFIG["SEED"], stratify=y_train
    )
    
    print(f"📊 Train: {len(X_train):,} samples")
    print(f"📊 Validation: {len(X_val):,} samples")
    print(f"📊 Test: {len(X_test):,} samples")
    
    # Load tokenizer and encoder
    print("🔄 Loading tokenizer and encoder...")
    tokenizer = BertTokenizerFast.from_pretrained(CONFIG["MODEL_NAME"])
    encoder = TFBertModel.from_pretrained(CONFIG["MODEL_NAME"])
    encoder.trainable = False  # Freeze encoder for faster training
    
    # Create datasets
    print("🔄 Creating optimized datasets...")
    train_dataset = create_optimized_dataset_tf(X_train, y_train, X_num_train, tokenizer, encoder)
    val_dataset = create_optimized_dataset_tf(X_val, y_val, X_num_val, tokenizer, encoder)
    test_dataset = create_optimized_dataset_tf(X_test, y_test, X_num_test, tokenizer, encoder)
    
    # Build model
    text_dim = 768
    numeric_dim = X_num_train.shape[1] if X_num_train.shape[1] > 0 else 0
    input_dim = text_dim + numeric_dim
    num_classes = len(le.classes_)
    
    print(f"🏗️  Building model...")
    print(f"   Input dimension: {input_dim}")
    print(f"   Text features: {text_dim}")
    print(f"   Numeric features: {numeric_dim}")
    print(f"   Number of classes: {num_classes}")
    
    model = create_optimized_model(input_dim, num_classes, text_dim, numeric_dim, strategy)
    
    # Train model
    trained_model, history = train_optimized_model(model, train_dataset, val_dataset, num_classes, strategy)
    
    # Evaluate model
    metrics = evaluate_model(trained_model, test_dataset)
    
    # Save model
    print("💾 Saving model...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save model components
    trained_model.save_weights(os.path.join(args.output_dir, 'model_weights.h5'))
    tokenizer.save_pretrained(os.path.join(args.output_dir, 'tokenizer'))
    encoder.save_pretrained(os.path.join(args.output_dir, 'encoder'))
    
    # Save bundle
    bundle = {
        'model_weights': os.path.join(args.output_dir, 'model_weights.h5'),
        'tokenizer_path': os.path.join(args.output_dir, 'tokenizer'),
        'encoder_path': os.path.join(args.output_dir, 'encoder'),
        'label_encoder': le,
        'scaler': scaler if numeric_cols else None,
        'numeric_cols': numeric_cols,
        'config': CONFIG,
        'metrics': metrics
    }
    
    joblib.dump(bundle, os.path.join(args.output_dir, 'model_bundle.joblib'))
    
    print(f"✅ Model saved to {args.output_dir}")
    print(f"📊 Final metrics: {metrics}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)