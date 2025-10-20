#!/usr/bin/env python3
"""
Test script to verify GPU optimization and performance
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

# Set environment variables
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
import numpy as np
import pandas as pd
from transformers import BertTokenizerFast, TFBertModel

def test_gpu_availability():
    """Test GPU availability and configuration"""
    print("🔍 Testing GPU availability...")
    
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("❌ No GPU detected")
        return False
    
    print(f"✅ Found {len(gpus)} GPU(s)")
    
    # Test GPU memory
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled")
    except Exception as e:
        print(f"⚠️  GPU memory growth failed: {e}")
    
    # Test mixed precision
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✅ Mixed precision enabled")
    except Exception as e:
        print(f"⚠️  Mixed precision failed: {e}")
    
    return True

def test_model_loading():
    """Test model loading and basic functionality"""
    print("\n🔄 Testing model loading...")
    
    try:
        # Test tokenizer loading
        tokenizer = BertTokenizerFast.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        print("✅ Tokenizer loaded successfully")
        
        # Test encoder loading
        encoder = TFBertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        print("✅ Encoder loaded successfully")
        
        # Test basic functionality
        test_text = "fever and cough with headache"
        encoding = tokenizer(test_text, return_tensors='tf', truncation=True, padding=True)
        outputs = encoder(encoding, training=False)
        embeddings = outputs.last_hidden_state[:, 0, :]
        
        print(f"✅ Text encoding successful: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_training_speed():
    """Test training speed with a simple model"""
    print("\n🚀 Testing training speed...")
    
    try:
        # Create a simple test model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=(768,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create test data
        X_test = np.random.random((1000, 768)).astype(np.float32)
        y_test = np.random.randint(0, 10, 1000)
        
        # Test training speed
        start_time = time.time()
        
        model.fit(
            X_test, y_test,
            epochs=5,
            batch_size=32,
            verbose=0
        )
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f} seconds")
        print(f"⚡ Speed: {1000 / training_time:.2f} samples/second")
        
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False

def test_memory_usage():
    """Test memory usage and optimization"""
    print("\n💾 Testing memory usage...")
    
    try:
        import psutil
        
        # Get initial memory
        initial_memory = psutil.virtual_memory().used / (1024**3)
        print(f"Initial RAM usage: {initial_memory:.2f} GB")
        
        # Create large test data
        X_large = np.random.random((10000, 768)).astype(np.float32)
        y_large = np.random.randint(0, 10, 10000)
        
        # Test memory with large data
        peak_memory = psutil.virtual_memory().used / (1024**3)
        print(f"Peak RAM usage: {peak_memory:.2f} GB")
        print(f"Memory increase: {peak_memory - initial_memory:.2f} GB")
        
        # Test GPU memory if available
        if tf.config.list_physical_devices('GPU'):
            try:
                # Create a large tensor on GPU
                gpu_tensor = tf.constant(X_large)
                print("✅ GPU memory test successful")
            except Exception as e:
                print(f"⚠️  GPU memory test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def test_parallel_processing():
    """Test parallel processing capabilities"""
    print("\n🔄 Testing parallel processing...")
    
    try:
        import multiprocessing as mp
        from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
        
        # Test CPU cores
        cpu_count = mp.cpu_count()
        print(f"CPU cores available: {cpu_count}")
        
        # Test parallel text processing
        test_texts = [
            "fever and cough",
            "headache and nausea",
            "chest pain and shortness of breath",
            "abdominal pain and vomiting",
            "fatigue and weakness"
        ] * 100  # 500 texts total
        
        def process_text(text):
            return text.lower().strip()
        
        # Test thread pool
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(process_text, test_texts))
        thread_time = time.time() - start_time
        
        print(f"✅ Thread pool processing: {thread_time:.2f} seconds")
        print(f"⚡ Speed: {len(test_texts) / thread_time:.2f} texts/second")
        
        return True
        
    except Exception as e:
        print(f"❌ Parallel processing test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 GPU Optimization Test Suite")
    print("=" * 50)
    
    tests = [
        ("GPU Availability", test_gpu_availability),
        ("Model Loading", test_model_loading),
        ("Training Speed", test_training_speed),
        ("Memory Usage", test_memory_usage),
        ("Parallel Processing", test_parallel_processing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your system is ready for GPU-optimized training.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)