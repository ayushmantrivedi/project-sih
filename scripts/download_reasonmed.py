import pandas as pd
import os
import json
from datasets import load_dataset
from tqdm import tqdm

def process_reasonmed():
    """
    Downloads and processes the ReasonMed dataset for BERT classification training.
    Extracts (question, diagnosis) pairs.
    """
    print("🚀 Streaming lingshu-medical-mllm/ReasonMed from Hugging Face (first 1,000 samples)...")
    try:
        # Using streaming=True to avoid downloading the entire 1.2GB file
        dataset = load_dataset("lingshu-medical-mllm/ReasonMed", split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    processed_data = []
    limit = 1000
    count = 0

    for item in tqdm(dataset, desc="Streaming samples", total=limit):
        if count >= limit:
            break
        
        question = item.get("question", "").strip()
        answer_key = item.get("answer", "").strip()
        options = item.get("options", [])
        
        # ReasonMed often uses multiple choice format. 
        # We need the TEXT of the correct option.
        correct_answer = ""
        
        if isinstance(options, list):
            # Format: ["A. Heart failure", "B. Pneumonia", ...]
            for opt in options:
                if opt.startswith(f"{answer_key}."):
                    # Extract text after 'A. '
                    parts = opt.split(".", 1)
                    if len(parts) > 1:
                        correct_answer = parts[1].strip()
                    else:
                        correct_answer = opt.strip()
                    break
        elif isinstance(options, dict):
            # Format: {"A": "Heart failure", "B": "Pneumonia"}
            correct_answer = options.get(answer_key, "").strip()
            
        if not correct_answer and answer_key:
            # Fallback if answer is already the text
            correct_answer = answer_key

        if question and correct_answer:
            processed_data.append({
                "symptoms": question,
                "disease": correct_answer
            })
            count += 1

    if not processed_data:
        print("⚠️ No valid data extracted!")
        return

    # Convert to DataFrame
    df = pd.DataFrame(processed_data)
    
    # Save to CSV in the project root for the trainer to pick up
    output_path = "reasonmed_processed.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✅ Success! Saved {len(df):,} records to {output_path}")
    print(f"📝 First 5 samples:")
    print(df.head())

if __name__ == "__main__":
    process_reasonmed()
