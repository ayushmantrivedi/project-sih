#!/usr/bin/env python3
# file: sih_global_line_search.py
# Frozen XLM-R embeddings + classifier trained with global-vs-local-error line-search logic.
# Hardcoded dataset path as requested.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF INFO

import re
import random
import json
import gc
from typing import List, Optional, Tuple
import argparse

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertModel

# -----------------------------
# Config (edit if needed)
# -----------------------------
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
CSV_PATH = r"C:\Users\ayush\OneDrive\Desktop\augmented_synthetic_health_dataset.csv"  # hardcoded raw string
JSON_PATH = r"C:\Users\ayush\OneDrive\Desktop\diagnosis_data.json"  # JSON data path
MAX_LEN = 64
BATCH_SIZE = 32  # Increased for large dataset
NUM_EPOCHS = 20  # Reduced for faster training with large dataset
BASE_LR = 2e-5
SEED = 42
EN_STOPWORDS = set(["and","or","the","is","a","an","it","to","of","in","for","on","with"])
HI_STOPWORDS = set(["और","है","को","का","कि","में","से","यह","मैं","हैं","हे","का"])
PUNCT_REGEX = re.compile(r"[\-–—.,;:!?()\[\]{}\"'\\/]")

# -----------------------------
# Helpers
# -----------------------------
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    # Remove punctuation and digits
    text = PUNCT_REGEX.sub(" ", text)
    text = re.sub(r"\d+", " ", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)
    # Remove common medical stopwords
    medical_stopwords = set(["mg", "ml", "tablet", "dose", "day", "week", "month", "year"])
    tokens = text.split()
    tokens = [t for t in tokens if t not in medical_stopwords]
    return " ".join(tokens)

def detect_lang(text: str) -> str:
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    return "en"

def simple_tokenize(text: str, lang_hint: Optional[str] = None) -> str:
    text = normalize_text(text)
    tokens = text.split()
    out = []
    for t in tokens:
        # Remove language-specific stopwords and short tokens
        if lang_hint == "hi":
            if t in HI_STOPWORDS or len(t) < 3:
                continue
        elif lang_hint == "en":
            if t in EN_STOPWORDS or len(t) < 3:
                continue
        else:
            if t in EN_STOPWORDS or t in HI_STOPWORDS or len(t) < 3:
                continue
        out.append(t)
    return " ".join(out)

# -----------------------------
# Data prep and embedding precompute
# -----------------------------
def load_json_data(json_path: str) -> pd.DataFrame:
    """Load data from JSON file and convert to DataFrame with robust error handling"""
    try:
        print(f"🔄 Loading large JSON file: {json_path}")
        print("⏳ This may take a moment for 100,000 records...")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert JSON to DataFrame
        df = pd.DataFrame(data)
        print(f"📊 Raw JSON data shape: {df.shape}")
        print(f"📋 Available columns: {list(df.columns)}")
        
        # Robust column mapping - try different possible column names
        column_mapping = {
            'symptoms': ['symptoms', 'symptom', 'symptom_text', 'complaint', 'complaints', 'description', 'desc'],
            'disease': ['disease', 'diagnosis', 'condition', 'illness', 'disorder', 'label', 'target', 'outcome']
        }
        
        # Find the best matching columns
        found_symptoms = None
        found_disease = None
        
        for col in df.columns:
            col_lower = col.lower().strip()
            for symptom_variant in column_mapping['symptoms']:
                if symptom_variant in col_lower or col_lower in symptom_variant:
                    found_symptoms = col
                    break
            if found_symptoms:
                break
        
        for col in df.columns:
            col_lower = col.lower().strip()
            for disease_variant in column_mapping['disease']:
                if disease_variant in col_lower or col_lower in disease_variant:
                    found_disease = col
                    break
            if found_disease:
                break
        
        # Handle missing required columns
        if not found_symptoms and not found_disease:
            print("⚠️  No suitable columns found for symptoms or disease")
            print("💡 Available columns:", list(df.columns))
            print("🔄 Attempting to use first two text columns...")
            
            # Try to use first two text columns
            text_cols = []
            for col in df.columns:
                if df[col].dtype == 'object':  # Text columns
                    text_cols.append(col)
            
            if len(text_cols) >= 2:
                found_symptoms = text_cols[0]
                found_disease = text_cols[1]
                print(f"✅ Using '{found_symptoms}' as symptoms and '{found_disease}' as disease")
            else:
                print("❌ Not enough text columns found. Skipping JSON data.")
                return pd.DataFrame()
        
        elif not found_symptoms:
            print("⚠️  No symptoms column found. Looking for alternative...")
            # Use description or first text column as symptoms
            text_cols = [col for col in df.columns if df[col].dtype == 'object' and col != found_disease]
            if text_cols:
                found_symptoms = text_cols[0]
                print(f"✅ Using '{found_symptoms}' as symptoms")
            else:
                print("❌ No suitable symptoms column found. Skipping JSON data.")
                return pd.DataFrame()
        
        elif not found_disease:
            print("⚠️  No disease column found. Looking for alternative...")
            # Use last text column or create dummy disease
            text_cols = [col for col in df.columns if df[col].dtype == 'object' and col != found_symptoms]
            if text_cols:
                found_disease = text_cols[-1]
                print(f"✅ Using '{found_disease}' as disease")
            else:
                print("⚠️  No disease column found. Creating 'unknown' disease for all records...")
                df['disease'] = 'unknown'
                found_disease = 'disease'
        
        # Rename columns to standard format
        df_standardized = df.copy()
        if found_symptoms != 'symptoms':
            df_standardized['symptoms'] = df[found_symptoms]
            print(f"🔄 Renamed '{found_symptoms}' → 'symptoms'")
        
        if found_disease != 'disease':
            df_standardized['disease'] = df[found_disease]
            print(f"🔄 Renamed '{found_disease}' → 'disease'")
        
        # Clean and validate the data
        df_standardized = df_standardized.dropna(subset=['symptoms', 'disease'])
        df_standardized['symptoms'] = df_standardized['symptoms'].astype(str)
        df_standardized['disease'] = df_standardized['disease'].astype(str)
        
        # Remove empty or very short entries
        df_standardized = df_standardized[
            (df_standardized['symptoms'].str.len() > 3) & 
            (df_standardized['disease'].str.len() > 1)
        ]
        
        print(f"✅ Successfully loaded and processed {len(df_standardized):,} records from JSON file")
        print(f"📊 Final JSON data shape: {df_standardized.shape}")
        print(f"📋 Final columns: {list(df_standardized.columns)}")
        print(f"🎯 Disease distribution: {df_standardized['disease'].value_counts().head()}")
        
        return df_standardized
        
    except Exception as e:
        print(f"❌ Error loading JSON file {json_path}: {e}")
        print("💡 The model will continue with CSV data only")
        return pd.DataFrame()

def load_and_prepare(csv_path: str, json_path: str = None):
    # Load CSV data with robust error handling
    try:
        df_csv = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df_csv):,} records from CSV file: {csv_path}")
        print(f"📋 CSV columns: {list(df_csv.columns)}")
        
        # Check for required columns in CSV
        if "symptoms" not in df_csv.columns or "disease" not in df_csv.columns:
            print("⚠️  CSV missing required columns. Attempting to find alternatives...")
            
            # Try to find alternative column names
            found_symptoms = None
            found_disease = None
            
            for col in df_csv.columns:
                col_lower = col.lower().strip()
                if any(x in col_lower for x in ['symptom', 'complaint', 'description']):
                    found_symptoms = col
                if any(x in col_lower for x in ['disease', 'diagnosis', 'condition', 'label']):
                    found_disease = col
            
            if found_symptoms and found_disease:
                df_csv['symptoms'] = df_csv[found_symptoms]
                df_csv['disease'] = df_csv[found_disease]
                print(f"✅ Mapped '{found_symptoms}' → 'symptoms' and '{found_disease}' → 'disease'")
            else:
                raise ValueError("CSV file must contain 'symptoms' and 'disease' columns or suitable alternatives")
        
        print(f"✅ CSV data validated: {len(df_csv):,} records")
        
    except Exception as e:
        print(f"❌ Error loading CSV file {csv_path}: {e}")
        raise SystemExit(f"Failed to load CSV data: {e}")
    
    # Load JSON data if provided
    df_json = pd.DataFrame()
    if json_path and os.path.exists(json_path):
        df_json = load_json_data(json_path)
    
    # Combine datasets
    if not df_json.empty:
        print(f"🔄 Combining datasets...")
        print(f"📊 CSV data: {len(df_csv):,} records")
        print(f"📊 JSON data: {len(df_json):,} records")
        
        # Ensure both datasets have the same columns
        common_cols = set(df_csv.columns) & set(df_json.columns)
        print(f"🔗 Common columns: {list(common_cols)}")
        
        df_csv = df_csv[list(common_cols)]
        df_json = df_json[list(common_cols)]
        
        # Combine the datasets
        print("⏳ Concatenating large datasets...")
        df = pd.concat([df_csv, df_json], ignore_index=True)
        print(f"✅ Combined dataset: {len(df_csv):,} CSV + {len(df_json):,} JSON = {len(df):,} total records")
        print(f"🎯 Total training data: {len(df):,} records")
    else:
        df = df_csv
        print(f"✅ Using only CSV data: {len(df):,} records")

    # Prepare text columns (but don't process yet - this will be done after train/test split)
    text_cols = ["symptoms"]
    if "description" in df.columns:
        text_cols.append("description")
    df["text_comb"] = df[text_cols].fillna("").agg(" ".join, axis=1)
    
    # Identify numeric features
    exclude = set(text_cols + ["text_comb", "disease"])
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    
    # Fill missing numeric values with 0
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(0.0)

    print(f"📊 Raw dataset shape: {df.shape}")
    print(f"📋 Numeric columns: {numeric_cols}")
    print(f"🎯 Disease distribution: {df['disease'].value_counts().head()}")
    
    return df, numeric_cols

def process_text_data(df, augment=True):
    """Process text data with augmentation (only for training data)"""
    # Process text
    df["text_proc"] = df["text_comb"].apply(lambda x: simple_tokenize(x, detect_lang(x))).astype(str)
    
    if augment:
        # Data augmentation only for training data
        synonym_dict = {
            "fever": ["pyrexia", "high temperature"],
            "cough": ["hack", "throat clearing"],
            "headache": ["cephalalgia", "head pain"],
            "pain": ["ache", "discomfort"],
            "nausea": ["queasiness", "sickness"],
            "vomiting": ["emesis", "throwing up"],
            "rash": ["eruption", "skin spots"],
            "chills": ["shivering", "cold sensation"]
        }

        def augment_text(text):
            words = text.split()
            aug_texts = []
            for i, w in enumerate(words):
                if w in synonym_dict:
                    for syn in synonym_dict[w]:
                        aug = words.copy()
                        aug[i] = syn
                        aug_texts.append(" ".join(aug))
            return aug_texts

        # Augment the dataframe
        aug_rows = []
        for idx, row in df.iterrows():
            aug_texts = augment_text(row["text_proc"])
            for aug in aug_texts:
                new_row = row.copy()
                new_row["text_proc"] = aug
                aug_rows.append(new_row)
        
        if aug_rows:
            df = pd.concat([df, pd.DataFrame(aug_rows)], ignore_index=True)
            print(f"✅ Added {len(aug_rows)} augmented samples")
    
    return df

def precompute_encoder_embeddings(tokenizer, encoder, texts: List[str], max_len: int = MAX_LEN, batch_size: int = 16) -> np.ndarray:
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="tf")
        outputs = encoder(enc, training=False)
        emb = outputs.last_hidden_state[:, 0, :].numpy()  # CLS embedding
        all_embs.append(emb)
    if all_embs:
        return np.vstack(all_embs)
    return np.zeros((0, encoder.config.hidden_size), dtype=np.float32)

# -----------------------------
# Classifier builder
# -----------------------------
def build_classifier(input_dim: int, num_classes: int, dropout: float = 0.3) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(input_dim,), dtype=tf.float32, name="fusion_input")

    # Attention mechanism for feature fusion
    # Assume first 768 are text features, rest are numeric
    text_dim = 768
    num_dim = input_dim - text_dim
    text_feats = tf.keras.layers.Lambda(lambda x: x[:, :text_dim])(inp)
    num_feats = tf.keras.layers.Lambda(lambda x: x[:, text_dim:])(inp)
    # Expand dims for attention
    text_exp = tf.keras.layers.Reshape((text_dim, 1))(text_feats)
    num_exp = tf.keras.layers.Reshape((num_dim, 1))(num_feats) if num_dim > 0 else None
    # Concatenate for attention
    if num_exp is not None:
        fusion = tf.keras.layers.Concatenate(axis=1)([text_exp, num_exp])
    else:
        fusion = text_exp
    # Simple attention: learn weights for each feature
    attn = tf.keras.layers.Dense(1, activation="softmax")(fusion)
    attn_out = tf.keras.layers.Multiply()([fusion, attn])
    attn_flat = tf.keras.layers.Flatten()(attn_out)

    x = tf.keras.layers.Dense(input_dim, activation="relu")(attn_flat)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(max(256, input_dim//2), activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    logits = tf.keras.layers.Dense(num_classes, name="logits")(x)  # logits, from_logits=True
    model = tf.keras.Model(inputs=inp, outputs=logits, name="fusion_classifier")
    return model

# -----------------------------
# Utility evaluation on fused numpy arrays
# -----------------------------
def eval_on_fused_numpy(classifier: tf.keras.Model, X_fused: np.ndarray, y: np.ndarray, loss_fn, batch_size: int = 256):
    n = len(y)
    losses = []
    preds = []
    trues = []
    for i in range(0, n, batch_size):
        xb = X_fused[i:i+batch_size]
        yb = y[i:i+batch_size]
        logits = classifier(xb, training=False).numpy()
        # compute loss
        loss_val = float(loss_fn(yb, logits).numpy())
        losses.append(loss_val)
        pb = np.argmax(logits, axis=1).tolist()
        preds.extend(pb)
        trues.extend(yb.tolist())
    avg_loss = float(np.mean(losses)) if losses else 0.0
    acc = float(accuracy_score(trues, preds)) if trues else 0.0
    f1 = float(f1_score(trues, preds, average="macro")) if trues else 0.0
    return avg_loss, acc, f1, preds, trues

# -----------------------------
# Training loop with global/local line-search
# -----------------------------
def train_with_global_local_line_search(classifier: tf.keras.Model,
                                        train_fused: np.ndarray, train_y: np.ndarray,
                                        val_fused: np.ndarray, val_y: np.ndarray,
                                        num_epochs: int = NUM_EPOCHS,
                                        base_lr: float = BASE_LR,
                                        batch_size: int = BATCH_SIZE,
                                        save_path: str = "best_classifier_weights.h5"):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # Use Adam optimizer; allow weight decay if TF supports AdamW
    try:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=base_lr, weight_decay=1e-6)
    except Exception:
        optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)

    best_val_loss = float("inf")
    best_weights = None
    best_opt_weights = None
    n_train = train_fused.shape[0]
    steps_per_epoch = max(1, n_train // batch_size)

    # Early stopping parameters
    patience = 5
    wait = 0
    min_delta = 1e-4

    for epoch in range(num_epochs):
        # shuffle training set each epoch
        perm = np.random.permutation(n_train)
        train_fused = train_fused[perm]
        train_y = train_y[perm]

        running_loss = 0.0
        steps = 0

        for i in range(0, n_train, batch_size):
            xb = train_fused[i:i+batch_size]
            yb = train_y[i:i+batch_size]
            steps += 1

            with tf.GradientTape() as tape:
                logits = classifier(xb, training=True)
                loss_value = loss_fn(yb, logits)
            grads = tape.gradient(loss_value, classifier.trainable_variables)
            local_error = float(loss_value.numpy())

            # compute current global (validation) error once per batch (can be expensive)
            val_loss, val_acc, val_f1, _, _ = eval_on_fused_numpy(classifier, val_fused, val_y, loss_fn)
            global_error = float(val_loss)

            # line-search style: if global_error < local_error try smaller learning-rate steps
            if global_error < local_error:
                weights_backup = classifier.get_weights()
                try:
                    opt_backup = optimizer.get_weights()
                except Exception:
                    opt_backup = None
                improved = False
                for k in range(4):
                    test_lr = float(base_lr * (0.5 ** (k + 1)))
                    try:
                        optimizer.learning_rate.assign(test_lr)
                    except Exception:
                        try:
                            optimizer = type(optimizer)(learning_rate=test_lr)
                        except Exception:
                            optimizer = tf.keras.optimizers.Adam(learning_rate=test_lr)
                        if opt_backup is not None:
                            try:
                                optimizer.set_weights(opt_backup)
                            except Exception:
                                pass
                    try:
                        optimizer.apply_gradients(zip(grads, classifier.trainable_variables))
                    except Exception:
                        classifier.set_weights(weights_backup)
                        if opt_backup is not None:
                            try:
                                optimizer.set_weights(opt_backup)
                            except Exception:
                                pass
                        continue
                    new_val_loss, _, _, _, _ = eval_on_fused_numpy(classifier, val_fused, val_y, loss_fn)
                    if new_val_loss < global_error:
                        improved = True
                        global_error = new_val_loss
                        break
                    else:
                        classifier.set_weights(weights_backup)
                        if opt_backup is not None:
                            try:
                                optimizer.set_weights(opt_backup)
                            except Exception:
                                pass
                if not improved:
                    try:
                        optimizer.learning_rate.assign(base_lr)
                    except Exception:
                        try:
                            optimizer = type(optimizer)(learning_rate=base_lr)
                        except Exception:
                            optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)
                        if opt_backup is not None:
                            try:
                                optimizer.set_weights(opt_backup)
                            except Exception:
                                pass
                    optimizer.apply_gradients(zip(grads, classifier.trainable_variables))
            else:
                optimizer.apply_gradients(zip(grads, classifier.trainable_variables))
            running_loss += local_error

        avg_train_loss = running_loss / max(1, steps)
        val_loss_epoch, val_acc_epoch, val_f1_epoch, _, _ = eval_on_fused_numpy(classifier, val_fused, val_y, loss_fn)
        print(f"Epoch {epoch+1}/{num_epochs} | TrainLoss: {avg_train_loss:.6f} | ValLoss: {val_loss_epoch:.6f} | ValAcc: {val_acc_epoch:.4f} | ValF1: {val_f1_epoch:.4f}")

        # Early stopping check
        if val_loss_epoch < best_val_loss - min_delta:
            best_val_loss = val_loss_epoch
            best_weights = classifier.get_weights()
            try:
                best_opt_weights = optimizer.get_weights()
            except Exception:
                best_opt_weights = None
            classifier.save_weights(save_path)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
                break

    # restore best
    if best_weights is not None:
        classifier.set_weights(best_weights)
        if best_opt_weights is not None:
            try:
                optimizer.set_weights(best_opt_weights)
            except Exception:
                pass
    return classifier

# -----------------------------
# Main pipeline
# -----------------------------
def main():
    set_seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=CSV_PATH)
    parser.add_argument("--json", type=str, default=JSON_PATH)
    parser.add_argument("--out_weights", type=str, default="best_classifier_weights.h5")
    parser.add_argument("--out_bundle", type=str, default="sih_model_bundle.joblib")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=BASE_LR)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")

    print("Loading combined dataset (CSV + JSON)...")
    df, numeric_cols = load_and_prepare(args.csv, args.json)
    assert "symptoms" in df.columns and "disease" in df.columns, "Combined dataset must contain 'symptoms' and 'disease' columns"

    # load tokenizer + encoder
    print("Loading tokenizer and TF encoder (this will download model if not present)...")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    encoder = TFBertModel.from_pretrained(MODEL_NAME)
    encoder.trainable = True

    # --- K-Fold Cross-Validation ---
    from sklearn.model_selection import StratifiedKFold
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    fold_metrics = []
    labels = df["disease"].astype(str).values
    y_all = LabelEncoder().fit_transform(labels)

    for fold, (train_idx, test_idx) in enumerate(skf.split(df, y_all)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")
        # Split data
        df_train = df.iloc[train_idx].copy()
        df_test = df.iloc[test_idx].copy()
        
        print(f"📊 Train: {len(df_train):,} records, Test: {len(df_test):,} records")

        # Process text data AFTER splitting to prevent data leakage
        print("🔄 Processing training data...")
        df_train = process_text_data(df_train, augment=True)
        
        print("🔄 Processing test data...")
        df_test = process_text_data(df_test, augment=False)  # No augmentation for test data

        # Numeric features
        exclude = set(text_cols + ["text_comb", "text_proc", "disease"])
        numeric_cols = [c for c in df_train.columns if c not in exclude and pd.api.types.is_numeric_dtype(df_train[c])]
        # Scaling: fit only on train, transform both
        if numeric_cols:
            scaler = StandardScaler()
            X_num_train = scaler.fit_transform(df_train[numeric_cols].values.astype(np.float32))
            X_num_test = scaler.transform(df_test[numeric_cols].values.astype(np.float32))
        else:
            X_num_train = np.zeros((len(df_train), 0), dtype=np.float32)
            X_num_test = np.zeros((len(df_test), 0), dtype=np.float32)

        # Labels - fit encoder only on training data to prevent leakage
        le = LabelEncoder()
        y_train = le.fit_transform(df_train["disease"].astype(str))
        y_test = le.transform(df_test["disease"].astype(str))
        
        print(f"🎯 Training classes: {len(le.classes_)}")
        print(f"📊 Class distribution in train: {pd.Series(y_train).value_counts().head()}")


        # --- Data Leakage Check ---
        print(f"[Data Leakage Check] Processing {len(df_train):,} train and {len(df_test):,} test records...")
        
        # Check for exact text overlap between train and test
        try:
            train_texts = set(df_train["text_proc"])
            test_texts = set(df_test["text_proc"])
            exact_overlap = train_texts.intersection(test_texts)
            print(f"🚨 EXACT TEXT OVERLAP: {len(exact_overlap)} samples")
            if len(exact_overlap) > 0:
                print("⚠️  WARNING: Data leakage detected! Some test samples appear in training data.")
                print(f"   Overlapping samples: {list(exact_overlap)[:5]}...")
            else:
                print("✅ No exact text overlap detected")
        except Exception as e:
            print(f"[Data Leakage Check] Text overlap check failed: {e}")
        
        # Memory-efficient duplicate checking
        try:
            train_dupes = df_train.duplicated(subset=["text_proc"]).sum()
            test_dupes = df_test.duplicated(subset=["text_proc"]).sum()
            print(f"[Diagnostics] Train duplicates: {train_dupes}, Test duplicates: {test_dupes}")
        except Exception as e:
            print(f"[Diagnostics] Duplicate check skipped due to memory constraints: {e}")
            train_dupes = test_dupes = 0
        
        # Memory-efficient overlap checking (sample-based for large datasets)
        try:
            if len(df_train) > 10000 or len(df_test) > 10000:
                # Sample-based overlap check for large datasets
                sample_size = min(5000, len(df_train), len(df_test))
                train_sample = df_train.sample(n=sample_size, random_state=42)
                test_sample = df_test.sample(n=sample_size, random_state=42)
                overlap = pd.merge(train_sample, test_sample, on=["text_proc"], how="inner")
                estimated_overlap = len(overlap) * (len(df_train) * len(df_test)) / (sample_size * sample_size)
                print(f"[Diagnostics] Estimated overlap: ~{int(estimated_overlap):,} (based on {sample_size:,} sample)")
            else:
                overlap = pd.merge(df_train, df_test, on=["text_proc"], how="inner")
                print(f"[Diagnostics] Overlap: {len(overlap)}")
        except Exception as e:
            print(f"[Diagnostics] Overlap check skipped due to memory constraints: {e}")
        
        # Class distribution (memory efficient)
        try:
            train_class_counts = df_train["disease"].value_counts()
            test_class_counts = df_test["disease"].value_counts()
            print(f"[Diagnostics] Train classes: {len(train_class_counts)}, Test classes: {len(test_class_counts)}")
            print(f"[Diagnostics] Top 5 train classes: {train_class_counts.head().to_dict()}")
            print(f"[Diagnostics] Top 5 test classes: {test_class_counts.head().to_dict()}")
        except Exception as e:
            print(f"[Diagnostics] Class distribution check failed: {e}")

        # Training and evaluation for this fold
        texts_train = df_train["text_proc"].tolist()
        texts_test = df_test["text_proc"].tolist()
        
        # Memory optimization for large datasets
        batch_size_emb = 8 if len(texts_train) > 50000 else 16
        
        print(f"Precomputing encoder embeddings for train set ({len(texts_train):,} records)...")
        print(f"Using batch size: {batch_size_emb} for memory optimization")
        emb_train = precompute_encoder_embeddings(tokenizer, encoder, texts_train, max_len=MAX_LEN, batch_size=batch_size_emb)
        
        print(f"Precomputing encoder embeddings for test set ({len(texts_test):,} records)...")
        emb_test = precompute_encoder_embeddings(tokenizer, encoder, texts_test, max_len=MAX_LEN, batch_size=batch_size_emb)

        def fuse(emb, num):
            if num is None or num.shape[-1] == 0:
                return emb
            return np.concatenate([emb, num.astype(np.float32)], axis=1)

        train_fused = fuse(emb_train, X_num_train)
        test_fused = fuse(emb_test, X_num_test)
        num_classes = len(le.classes_)
        input_dim = train_fused.shape[1]
        classifier = build_classifier(input_dim, num_classes, dropout=0.3)
        _ = classifier(np.zeros((1, input_dim), dtype=np.float32))
        trained_classifier = train_with_global_local_line_search(
            classifier,
            train_fused, y_train,
            test_fused, y_test,
            num_epochs=args.epochs,
            base_lr=args.lr,
            batch_size=args.batch_size,
            save_path=args.out_weights
        )
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        test_loss, test_acc, test_f1, test_preds, test_trues = eval_on_fused_numpy(trained_classifier, test_fused, y_test, loss_fn)
        print(f"Fold {fold+1} | Loss: {test_loss:.6f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
        fold_metrics.append((test_loss, test_acc, test_f1))
        
        # Memory cleanup after each fold
        del df_train, df_test, texts_train, texts_test, emb_train, emb_test, train_fused, test_fused
        del trained_classifier, classifier
        gc.collect()
        print(f"🧹 Memory cleaned after fold {fold+1}")

    # Report average metrics
    avg_loss = np.mean([m[0] for m in fold_metrics])
    avg_acc = np.mean([m[1] for m in fold_metrics])
    avg_f1 = np.mean([m[2] for m in fold_metrics])
    print(f"\nK-Fold CV Results | Avg Loss: {avg_loss:.6f} | Avg Acc: {avg_acc:.4f} | Avg F1: {avg_f1:.4f}")

    # Save artifacts: classifier weights, tokenizer, encoder, label encoder, scaler, numeric_cols
    tk_dir = "saved_tokenizer"
    enc_dir = "saved_encoder"
    os.makedirs(tk_dir, exist_ok=True)
    os.makedirs(enc_dir, exist_ok=True)
    tokenizer.save_pretrained(tk_dir)
    encoder.save_pretrained(enc_dir)
    trained_classifier.save_weights(args.out_weights)

    # Use the label encoder from the last fold (they should all be the same)
    bundle = {
        "classifier_weights": os.path.abspath(args.out_weights),
        "label_encoder": le,
        "tokenizer_dir": os.path.abspath(tk_dir),
        "encoder_dir": os.path.abspath(enc_dir),
        "numeric_cols": numeric_cols,
        "scaler": scaler
    }
    joblib.dump(bundle, args.out_bundle)
    print("Saved classifier weights to", args.out_weights)
    print("Saved bundle to", args.out_bundle)
    print("Label mapping:")
    for lbl, encv in zip(le.classes_, le.transform(le.classes_)):
        print(lbl, "->", encv)

    # Demo predict helper using precomputed encoder for a single text + numeric vector
    def predict_new_local(text: str, numeric_feats: Optional[List[float]] = None):
        tproc = simple_tokenize(text, detect_lang(text))
        enc = tokenizer([tproc], truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="tf")
        outputs = encoder(enc, training=False)
        emb = outputs.last_hidden_state[:, 0, :].numpy()
        if numeric_feats is None or len(numeric_feats) == 0:
            fused = emb
        else:
            nf = np.array(numeric_feats, dtype=np.float32).reshape(1, -1)
            fused = np.concatenate([emb, nf], axis=1)
        logits = trained_classifier(fused, training=False).numpy()
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
        idx = int(np.argmax(probs))
        label = le.inverse_transform([idx])[0]
        return label, float(np.max(probs)), probs

    # demo
    demo_text = "fever and cough with headache"
    # Ensure demo_numeric matches the length of numeric_cols
    demo_numeric = [0.0] * len(numeric_cols) if numeric_cols else None
    lbl, conf, probs = predict_new_local(demo_text, demo_numeric)
    print(f"\nDemo: '{demo_text}' -> {lbl} (conf {conf:.3f})")

# Global model variables (will be loaded when needed)
_global_tokenizer = None
_global_encoder = None
_global_classifier = None
_global_label_encoder = None
_global_scaler = None
_global_numeric_cols = None

def load_trained_model(model_path: str = "trained_health_model.joblib"):
    """Load the trained model components"""
    global _global_tokenizer, _global_encoder, _global_classifier, _global_label_encoder, _global_scaler, _global_numeric_cols
    
    try:
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found. Please train the model first.")
            return False
        
        bundle = joblib.load(model_path)
        _global_tokenizer = bundle["tokenizer"]
        _global_encoder = bundle["encoder"]
        _global_classifier = bundle["classifier"]
        _global_label_encoder = bundle["label_encoder"]
        _global_scaler = bundle.get("scaler", None)
        _global_numeric_cols = bundle.get("numeric_cols", [])
        
        print(f"✅ Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def predict_new(text: str, numeric_feats: Optional[List[float]] = None):
    """
    Predict disease from text symptoms using the trained model
    
    Args:
        text: Symptom description
        numeric_feats: Optional numeric features
    
    Returns:
        tuple: (predicted_label, confidence, probabilities)
    """
    global _global_tokenizer, _global_encoder, _global_classifier, _global_label_encoder, _global_scaler, _global_numeric_cols
    
    # Check if model is loaded
    if _global_tokenizer is None:
        print("⚠️ Model not loaded. Loading now...")
        if not load_trained_model():
            return "Model not available", 0.0, []
    
    try:
        # Process text
        tproc = simple_tokenize(text, detect_lang(text))
        enc = _global_tokenizer([tproc], truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="tf")
        
        # Get embeddings
        outputs = _global_encoder(enc, training=False)
        emb = outputs.last_hidden_state[:, 0, :].numpy()
        
        # Handle numeric features
        if numeric_feats is None or len(numeric_feats) == 0:
            fused = emb
        else:
            nf = np.array(numeric_feats, dtype=np.float32).reshape(1, -1)
            if _global_scaler is not None:
                nf = _global_scaler.transform(nf)
            fused = np.concatenate([emb, nf], axis=1)
        
        # Get predictions
        logits = _global_classifier(fused, training=False).numpy()
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
        idx = int(np.argmax(probs))
        label = _global_label_encoder.inverse_transform([idx])[0]
        
        return label, float(np.max(probs)), probs
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return "Error", 0.0, []

if __name__ == "__main__":
    main()
