#!/usr/bin/env python3
"""
Ultra-Fast GPU Training Script for HealthBot AI Chatbot
Features:
- Multi-GPU support
- Mixed precision training
- Data parallelism
- Memory optimization
- Parallel data loading
- Advanced optimizations
"""

import os
import sys
import argparse
import time
import gc
import psutil
from typing import List, Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# Set environment variables for maximum performance
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_GPU_THREAD_COUNT"] = "2"

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertModel
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import re

# Ultra-optimized configuration
ULTRA_CONFIG = {
    "MODEL_NAME": "emilyalsentzer/Bio_ClinicalBERT",
    "MAX_LEN": 128,
    "BATCH_SIZE": 128,  # Large batch for GPU
    "NUM_EPOCHS": 30,
    "BASE_LR": 5e-5,
    "SEED": 42,
    "MIXED_PRECISION": True,
    "USE_MULTI_GPU": True,
    "NUM_WORKERS": min(16, mp.cpu_count()),
    "PREFETCH_BUFFER": 8,
    "CACHE_DATASET": True,
    "EARLY_STOPPING_PATIENCE": 10,
    "LEARNING_RATE_SCHEDULE": True,
    "WEIGHT_DECAY": 1e-4,
    "GRADIENT_CLIPPING": 1.0,
    "WARMUP_STEPS": 2000,
    "USE_GRADIENT_CHECKPOINTING": True,
    "USE_XLA": True,  # Accelerated Linear Algebra
    "USE_AMP": True,  # Automatic Mixed Precision
    "OPTIMIZER": "adamw",
    "SCHEDULER": "cosine",
    "DROPOUT_RATE": 0.3,
    "LABEL_SMOOTHING": 0.1,
    "USE_FOCAL_LOSS": False,
    "FOCAL_ALPHA": 0.25,
    "FOCAL_GAMMA": 2.0,
}

def setup_ultra_gpu():
    """Setup ultra-optimized GPU configuration"""
    print("🚀 Setting up ultra-optimized GPU configuration...")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("⚠️  No GPU detected. Training will use CPU (very slow)")
        return False
    
    print(f"✅ Found {len(gpus)} GPU(s)")
    
    # Configure GPU memory growth and optimization
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # Enable memory optimization
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]
            )
        print("✅ GPU memory optimization enabled")
    except RuntimeError as e:
        print(f"⚠️  GPU memory optimization failed: {e}")
    
    # Enable mixed precision
    if ULTRA_CONFIG["MIXED_PRECISION"]:
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✅ Mixed precision training enabled")
        except Exception as e:
            print(f"⚠️  Mixed precision setup failed: {e}")
            ULTRA_CONFIG["MIXED_PRECISION"] = False
    
    # Setup multi-GPU strategy
    if ULTRA_CONFIG["USE_MULTI_GPU"] and len(gpus) > 1:
        try:
            strategy = tf.distribute.MirroredStrategy(
                cross_device_ops=tf.distribute.ReductionToOneDevice()
            )
            print(f"✅ Multi-GPU strategy enabled with {strategy.num_replicas_in_sync} GPUs")
            return strategy
        except Exception as e:
            print(f"⚠️  Multi-GPU setup failed: {e}")
            ULTRA_CONFIG["USE_MULTI_GPU"] = False
    
    # Enable XLA compilation
    if ULTRA_CONFIG["USE_XLA"]:
        try:
            tf.config.optimizer.set_jit(True)
            print("✅ XLA compilation enabled")
        except Exception as e:
            print(f"⚠️  XLA setup failed: {e}")
            ULTRA_CONFIG["USE_XLA"] = False
    
    return True

def set_seed(seed: int = ULTRA_CONFIG["SEED"]):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def normalize_text_ultra(text: str) -> str:
    """Ultra-optimized text normalization"""
    if not isinstance(text, str):
        text = str(text)
    
    text = text.lower().strip()
    
    # Fast regex operations
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Medical stopwords (pre-compiled for speed)
    medical_stopwords = {
        "mg", "ml", "tablet", "dose", "day", "week", "month", "year",
        "patient", "history", "present", "complaint", "chief", "case",
        "admission", "discharge", "follow", "up", "visit", "examination"
    }
    
    tokens = text.split()
    tokens = [t for t in tokens if t not in medical_stopwords and len(t) > 2]
    
    return " ".join(tokens)

def create_ultra_dataset(csv_path: str, json_path: Optional[str] = None, 
                        sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Create ultra-optimized dataset with maximum parallel processing"""
    print("📊 Creating ultra-optimized dataset...")
    
    def load_csv_ultra(csv_path: str) -> pd.DataFrame:
        """Ultra-fast CSV loading with chunking"""
        try:
            # Use optimal chunk size based on available memory
            memory_gb = psutil.virtual_memory().available / (1024**3)
            chunk_size = min(50000, int(memory_gb * 10000))
            
            chunks = []
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
                chunks.append(chunk)
                if len(chunks) * chunk_size > 200000:  # Limit memory usage
                    break
            
            df = pd.concat(chunks, ignore_index=True)
            print(f"✅ Loaded {len(df):,} records from CSV")
            return df
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return pd.DataFrame()
    
    def load_json_ultra(json_path: str) -> pd.DataFrame:
        """Ultra-fast JSON loading"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            print(f"✅ Loaded {len(df):,} records from JSON")
            return df
        except Exception as e:
            print(f"❌ Error loading JSON: {e}")
            return pd.DataFrame()
    
    # Load data with maximum parallelism
    with ThreadPoolExecutor(max_workers=4) as executor:
        csv_future = executor.submit(load_csv_ultra, csv_path)
        json_future = executor.submit(load_json_ultra, json_path) if json_path else None
        
        df_csv = csv_future.result()
        df_json = json_future.result() if json_future else pd.DataFrame()
    
    # Combine datasets efficiently
    if not df_json.empty:
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
        df = df.sample(n=sample_size, random_state=ULTRA_CONFIG["SEED"])
        print(f"📊 Sampled to {len(df):,} records")
    
    # Ultra-fast data cleaning
    print("🧹 Ultra-fast data cleaning...")
    
    # Find columns efficiently
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
    
    # Parallel text processing
    print("🔄 Parallel text processing...")
    with ThreadPoolExecutor(max_workers=ULTRA_CONFIG["NUM_WORKERS"]) as executor:
        processed_texts = list(executor.map(normalize_text_ultra, df[text_col].tolist()))
    
    df[text_col] = processed_texts
    
    # Clean data
    df = df.dropna(subset=[text_col, label_col])
    df = df[df[text_col].str.len() > 3]
    
    # Identify numeric features
    exclude = {text_col, label_col}
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    
    # Fill missing numeric values
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(0.0)
    
    print(f"✅ Ultra-optimized dataset: {len(df):,} records, {len(numeric_cols)} numeric features")
    
    return df, numeric_cols

def create_ultra_model(input_dim: int, num_classes: int, 
                      text_dim: int = 768, numeric_dim: int = 0,
                      strategy=None) -> tf.keras.Model:
    """Create ultra-optimized model architecture"""
    
    def build_model():
        inputs = tf.keras.Input(shape=(input_dim,), dtype=tf.float32, name="input")
        
        # Separate text and numeric features
        text_features = tf.keras.layers.Lambda(lambda x: x[:, :text_dim])(inputs)
        numeric_features = tf.keras.layers.Lambda(lambda x: x[:, text_dim:])(inputs) if numeric_dim > 0 else None
        
        # Ultra-optimized text processing branch
        text_branch = tf.keras.layers.Dense(2048, activation="gelu", name="text_dense1")(text_features)
        text_branch = tf.keras.layers.BatchNormalization()(text_branch)
        text_branch = tf.keras.layers.Dropout(ULTRA_CONFIG["DROPOUT_RATE"])(text_branch)
        
        text_branch = tf.keras.layers.Dense(1024, activation="gelu", name="text_dense2")(text_branch)
        text_branch = tf.keras.layers.BatchNormalization()(text_branch)
        text_branch = tf.keras.layers.Dropout(ULTRA_CONFIG["DROPOUT_RATE"])(text_branch)
        
        text_branch = tf.keras.layers.Dense(512, activation="gelu", name="text_dense3")(text_branch)
        text_branch = tf.keras.layers.BatchNormalization()(text_branch)
        text_branch = tf.keras.layers.Dropout(ULTRA_CONFIG["DROPOUT_RATE"] * 0.5)(text_branch)
        
        # Numeric processing branch
        if numeric_dim > 0:
            numeric_branch = tf.keras.layers.Dense(max(256, numeric_dim * 8), activation="gelu")(numeric_features)
            numeric_branch = tf.keras.layers.BatchNormalization()(numeric_branch)
            numeric_branch = tf.keras.layers.Dropout(ULTRA_CONFIG["DROPOUT_RATE"] * 0.5)(numeric_branch)
            
            # Advanced feature fusion with attention
            fused = tf.keras.layers.Concatenate()([text_branch, numeric_branch])
            
            # Multi-head attention for feature fusion
            attention = tf.keras.layers.MultiHeadAttention(
                num_heads=8, key_dim=64, dropout=0.1
            )(fused, fused)
            
            attended = tf.keras.layers.Add()([fused, attention])
            attended = tf.keras.layers.LayerNormalization()(attended)
        else:
            attended = text_branch
        
        # Final classification layers with residual connections
        x = tf.keras.layers.Dense(512, activation="gelu")(attended)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(ULTRA_CONFIG["DROPOUT_RATE"])(x)
        
        # Residual connection
        residual = tf.keras.layers.Dense(512, activation="gelu")(attended)
        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.LayerNormalization()(x)
        
        x = tf.keras.layers.Dense(256, activation="gelu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(ULTRA_CONFIG["DROPOUT_RATE"] * 0.5)(x)
        
        x = tf.keras.layers.Dense(128, activation="gelu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(ULTRA_CONFIG["DROPOUT_RATE"] * 0.3)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(num_classes, name="outputs")(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ultra_optimized_classifier")
        return model
    
    if strategy:
        with strategy.scope():
            return build_model()
    else:
        return build_model()

def create_ultra_optimizer(num_train_steps: int) -> tf.keras.optimizers.Optimizer:
    """Create ultra-optimized optimizer"""
    
    # Advanced learning rate schedule
    if ULTRA_CONFIG["SCHEDULER"] == "cosine":
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=ULTRA_CONFIG["BASE_LR"],
            first_decay_steps=num_train_steps // 4,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.1
        )
    else:
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=ULTRA_CONFIG["BASE_LR"],
            decay_steps=num_train_steps,
            end_learning_rate=ULTRA_CONFIG["BASE_LR"] * 0.01,
            power=0.5
        )
    
    # Create optimizer
    if ULTRA_CONFIG["OPTIMIZER"] == "adamw":
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=ULTRA_CONFIG["WEIGHT_DECAY"],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            clipnorm=ULTRA_CONFIG["GRADIENT_CLIPPING"]
        )
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            clipnorm=ULTRA_CONFIG["GRADIENT_CLIPPING"]
        )
    
    return optimizer

def create_ultra_dataset_tf(texts: List[str], labels: List[int], 
                           numeric_features: np.ndarray,
                           tokenizer, encoder, 
                           batch_size: int = ULTRA_CONFIG["BATCH_SIZE"]) -> tf.data.Dataset:
    """Create ultra-optimized TensorFlow dataset"""
    
    def process_text_batch(texts_batch):
        """Process batch of texts"""
        return [normalize_text_ultra(text) for text in texts_batch]
    
    def encode_text_batch(texts_batch):
        """Encode batch of texts"""
        encodings = tokenizer(
            texts_batch,
            truncation=True,
            padding='max_length',
            max_length=ULTRA_CONFIG["MAX_LEN"],
            return_tensors='tf'
        )
        
        outputs = encoder(encodings, training=False)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        return embeddings
    
    # Process texts in parallel batches
    print("🔄 Ultra-parallel text processing...")
    batch_size_process = 1000
    processed_texts = []
    
    for i in range(0, len(texts), batch_size_process):
        batch_texts = texts[i:i + batch_size_process]
        with ThreadPoolExecutor(max_workers=ULTRA_CONFIG["NUM_WORKERS"]) as executor:
            batch_processed = list(executor.map(normalize_text_ultra, batch_texts))
        processed_texts.extend(batch_processed)
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices({
        'text': processed_texts,
        'label': labels,
        'numeric': numeric_features
    })
    
    # Batch and encode texts
    dataset = dataset.batch(32)  # Smaller batch for encoding
    
    def encode_batch(x):
        text_emb = encode_text_batch(x['text'])
        return {
            'text_emb': text_emb,
            'label': x['label'],
            'numeric': x['numeric']
        }
    
    dataset = dataset.map(encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Fuse features
    def fuse_features(x):
        if x['numeric'].shape[1] > 0:
            fused = tf.concat([x['text_emb'], x['numeric']], axis=1)
        else:
            fused = x['text_emb']
        return fused, x['label']
    
    dataset = dataset.map(fuse_features, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Ultra-optimize dataset
    dataset = dataset.unbatch()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(ULTRA_CONFIG["PREFETCH_BUFFER"])
    
    if ULTRA_CONFIG["CACHE_DATASET"]:
        dataset = dataset.cache()
    
    return dataset

def train_ultra_model(model: tf.keras.Model, 
                     train_dataset: tf.data.Dataset,
                     val_dataset: tf.data.Dataset,
                     num_classes: int,
                     strategy=None) -> tf.keras.Model:
    """Train model with ultra-optimizations"""
    
    print("🚀 Starting ultra-optimized training...")
    
    # Calculate training steps
    num_train_steps = sum(1 for _ in train_dataset) * ULTRA_CONFIG["NUM_EPOCHS"]
    
    # Create optimizer
    optimizer = create_ultra_optimizer(num_train_steps)
    
    # Loss function with label smoothing
    if ULTRA_CONFIG["LABEL_SMOOTHING"] > 0:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            label_smoothing=ULTRA_CONFIG["LABEL_SMOOTHING"]
        )
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['sparse_categorical_accuracy']
    )
    
    # Ultra-optimized callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=ULTRA_CONFIG["EARLY_STOPPING_PATIENCE"],
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-8,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'ultra_best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    # Training
    print(f"📊 Ultra-training for {ULTRA_CONFIG['NUM_EPOCHS']} epochs...")
    print(f"📊 Batch size: {ULTRA_CONFIG['BATCH_SIZE']}")
    print(f"📊 Learning rate: {ULTRA_CONFIG['BASE_LR']}")
    print(f"📊 Mixed precision: {ULTRA_CONFIG['MIXED_PRECISION']}")
    print(f"📊 Multi-GPU: {ULTRA_CONFIG['USE_MULTI_GPU']}")
    
    start_time = time.time()
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=ULTRA_CONFIG["NUM_EPOCHS"],
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"⏱️  Ultra-training completed in {training_time:.2f} seconds")
    print(f"⚡ Speed: {num_train_steps / training_time:.2f} steps/second")
    
    return model, history

def main():
    """Main ultra-optimized training function"""
    parser = argparse.ArgumentParser(description='Ultra-Fast GPU Training')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV data')
    parser.add_argument('--json', type=str, default=None, help='Path to JSON data')
    parser.add_argument('--epochs', type=int, default=ULTRA_CONFIG["NUM_EPOCHS"])
    parser.add_argument('--batch_size', type=int, default=ULTRA_CONFIG["BATCH_SIZE"])
    parser.add_argument('--lr', type=float, default=ULTRA_CONFIG["BASE_LR"])
    parser.add_argument('--sample_size', type=int, default=None, help='Sample size for testing')
    parser.add_argument('--output_dir', type=str, default='ultra_optimized_model')
    
    args = parser.parse_args()
    
    # Update config
    ULTRA_CONFIG["NUM_EPOCHS"] = args.epochs
    ULTRA_CONFIG["BATCH_SIZE"] = args.batch_size
    ULTRA_CONFIG["BASE_LR"] = args.lr
    
    print("🚀 ULTRA-FAST GPU TRAINING")
    print("=" * 60)
    print(f"📊 Epochs: {ULTRA_CONFIG['NUM_EPOCHS']}")
    print(f"📊 Batch size: {ULTRA_CONFIG['BATCH_SIZE']}")
    print(f"📊 Learning rate: {ULTRA_CONFIG['BASE_LR']}")
    print(f"📊 Mixed precision: {ULTRA_CONFIG['MIXED_PRECISION']}")
    print(f"📊 Multi-GPU: {ULTRA_CONFIG['USE_MULTI_GPU']}")
    print(f"📊 XLA: {ULTRA_CONFIG['USE_XLA']}")
    print(f"📊 Workers: {ULTRA_CONFIG['NUM_WORKERS']}")
    print()
    
    # Setup
    set_seed()
    strategy = setup_ultra_gpu()
    
    # Load data
    df, numeric_cols = create_ultra_dataset(args.csv, args.json, args.sample_size)
    
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
        texts, y_encoded, X_numeric, test_size=0.2, random_state=ULTRA_CONFIG["SEED"], stratify=y_encoded
    )
    
    X_train, X_val, y_train, y_val, X_num_train, X_num_val = train_test_split(
        X_train, y_train, X_num_train, test_size=0.2, random_state=ULTRA_CONFIG["SEED"], stratify=y_train
    )
    
    print(f"📊 Train: {len(X_train):,} samples")
    print(f"📊 Validation: {len(X_val):,} samples")
    print(f"📊 Test: {len(X_test):,} samples")
    
    # Load tokenizer and encoder
    print("🔄 Loading tokenizer and encoder...")
    tokenizer = BertTokenizerFast.from_pretrained(ULTRA_CONFIG["MODEL_NAME"])
    encoder = TFBertModel.from_pretrained(ULTRA_CONFIG["MODEL_NAME"])
    encoder.trainable = False  # Freeze for speed
    
    # Create ultra-optimized datasets
    print("🔄 Creating ultra-optimized datasets...")
    train_dataset = create_ultra_dataset_tf(X_train, y_train, X_num_train, tokenizer, encoder)
    val_dataset = create_ultra_dataset_tf(X_val, y_val, X_num_val, tokenizer, encoder)
    test_dataset = create_ultra_dataset_tf(X_test, y_test, X_num_test, tokenizer, encoder)
    
    # Build ultra-optimized model
    text_dim = 768
    numeric_dim = X_num_train.shape[1] if X_num_train.shape[1] > 0 else 0
    input_dim = text_dim + numeric_dim
    num_classes = len(le.classes_)
    
    print(f"🏗️  Building ultra-optimized model...")
    print(f"   Input dimension: {input_dim}")
    print(f"   Text features: {text_dim}")
    print(f"   Numeric features: {numeric_dim}")
    print(f"   Number of classes: {num_classes}")
    
    model = create_ultra_model(input_dim, num_classes, text_dim, numeric_dim, strategy)
    
    # Train ultra-optimized model
    trained_model, history = train_ultra_model(model, train_dataset, val_dataset, num_classes, strategy)
    
    # Save ultra-optimized model
    print("💾 Saving ultra-optimized model...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save model components
    trained_model.save_weights(os.path.join(args.output_dir, 'ultra_model_weights.h5'))
    tokenizer.save_pretrained(os.path.join(args.output_dir, 'tokenizer'))
    encoder.save_pretrained(os.path.join(args.output_dir, 'encoder'))
    
    # Save bundle
    bundle = {
        'model_weights': os.path.join(args.output_dir, 'ultra_model_weights.h5'),
        'tokenizer_path': os.path.join(args.output_dir, 'tokenizer'),
        'encoder_path': os.path.join(args.output_dir, 'encoder'),
        'label_encoder': le,
        'scaler': scaler if numeric_cols else None,
        'numeric_cols': numeric_cols,
        'config': ULTRA_CONFIG,
        'training_history': history.history
    }
    
    joblib.dump(bundle, os.path.join(args.output_dir, 'ultra_model_bundle.joblib'))
    
    print(f"✅ Ultra-optimized model saved to {args.output_dir}")
    print("🎯 Your model is now ultra-optimized for maximum speed!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)