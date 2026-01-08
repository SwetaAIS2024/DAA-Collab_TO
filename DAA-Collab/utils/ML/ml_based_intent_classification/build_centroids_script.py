import argparse
import sys
import os
import pandas as pd
import numpy as np

import pandas as pd
from ast import literal_eval
import joblib
import os, sys
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(os.path.join(BASE_DIR, "DAA-Collab"))

DATA_PATH = os.path.join(BASE_DIR, "DAA-Collab/data/user_prompt_dataset/traffic_prompts_realistic_option2_FIXED.csv")

MODEL_DIR = os.path.join(BASE_DIR,"DAA-Collab/utils/ML/ml_based_intent_classification/saved_models_transformer")
os.makedirs(MODEL_DIR, exist_ok=True)

def build_intent_centroids_index(include_memory_data: bool = False):

    MODEL_NAME = 'Qwen/Qwen3-Embedding-8B'
    # Detect device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Load pre-trained sentence transformer (cache for offline use)
    try:
        # model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODEL_DIR) 
        model = SentenceTransformer(MODEL_NAME, cache_folder = MODEL_DIR, trust_remote_code=True, device=DEVICE)
        print("✅ Sentence Transformer model loaded on device:", DEVICE)
    except Exception as e:
        print(f"⚠️ Error loading Sentence Transformer: {e}")
        print("   Run: pip install sentence-transformers")
        model = None
    
    if model is None:
        raise RuntimeError("Sentence Transformer not available. Install: pip install sentence-transformers")
    

    print("\n" + "="*80)
    print("BUILDING INTENT CENTROIDS INDEX USING EMBEDDING MODEL")
    print("="*80)
    
    # Load original training data
    df = pd.read_csv(DATA_PATH)
    df["intents"] = df["intents"].apply(literal_eval)
    
    # Optionally merge with validated memory data
    if include_memory_data:
        from utils.memory.episodic import get_intent_memory
        memory = get_intent_memory()
        # FIX: For now, let's allow unvalidated interactions too if validated ones are scarce, 
        # or just be aware that validated_only=True is the default.
        # The user's issue is "no interactions", which means they probably haven't validated any yet.
        # If we want to train on *all* memory, we should set validated_only=False.
        # However, usually we only want to train on "good" data.
        # Let's try to get validated ones first, and if empty, maybe warn the user.
        
        print("DEBUG: Attempting to export validated interactions from memory...")
        memory_file = memory.export_to_training_dataset(validated_only=True)
        
        if not memory_file:
             print("DEBUG: No validated interactions found. Trying unvalidated for demonstration...")
             memory_file = memory.export_to_training_dataset(validated_only=False)

        if memory_file and os.path.exists(memory_file):
            df_memory = pd.read_csv(memory_file)
            
            # FIX: Drop rows with NaN intents before applying literal_eval
            # This prevents "malformed node or string: nan" error
            df_memory = df_memory.dropna(subset=['intents'])
            
            # Also ensure intents are strings before eval
            df_memory = df_memory[df_memory['intents'].apply(lambda x: isinstance(x, str))]
            
            df_memory["intents"] = df_memory["intents"].apply(literal_eval)
            
            # Merge datasets
            df = pd.concat([df, df_memory], ignore_index=True)
            print(f"✅ Included {len(df_memory)} interactions from memory")

    # Create intent prototypes (average embeddings per intent)
    intent_centroids = {}
    intent_examples = {}
    
    # Group prompts by intent
    for _, row in df.iterrows():
        for intent in row['intents']:
            if intent not in intent_examples:
                intent_examples[intent] = []
            intent_examples[intent].append(row['prompt'])
    
    print(f"\nTotal unique intents: {len(intent_examples)}")
    print(f"Total [user query - intent] samples: {len(df)}")
    print("Centroid embeddings for each class are built using all available samples per intent (up to 1000 samples each).")
    print("\nIntent distribution:")
    for intent, examples in sorted(intent_examples.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {intent}: {len(examples)} examples")
    
    print("\nCreating semantic embeddings...")
    
    # Create centroid embeddings for each intent
    for intent, examples in intent_examples.items():
        # Use top 1000 examples per intent (or all if less)
        sample_examples = examples[:1000]
        print(f"  Encoding {intent}: {len(sample_examples)} examples...")

        # changes HERE for using the GPU for the embedding computation
        embeddings = model.encode(
            sample_examples, 
            show_progress_bar=True,
            batch_size=32, # << HERE 
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=DEVICE # << HERE 
            )
        # Average embedding as centroid
        intent_centroids[intent] = np.mean(embeddings, axis=0)
    
    # Save  centroids and classes
    joblib.dump(intent_centroids, os.path.join(MODEL_DIR, "intent_centroids.joblib"))
    joblib.dump(list(intent_examples.keys()), os.path.join(MODEL_DIR, "intent_classes.joblib"))
    
    print(f"\n✅ Embeddings saved to: {MODEL_DIR}")
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="BUILDING QUERY INTENT CENTROID INDEX USING EMBEDDING MODEL")
    parser.add_argument(
        '--include-memory',
        action='store_true',
        help='Include validated interactions from LangGraph memory'
    )
    
    args = parser.parse_args()
    print(f"Include memory data: {args.include_memory}\n")
    
    try:
        build_intent_centroids_index(include_memory_data=args.include_memory)
        print("\n✅ Building process completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Building process failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()