import pandas as pd
from ast import literal_eval
import joblib
import os, sys
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.common import preprocess_prompt

MODEL_NAME = 'Alibaba-NLP/gte-large-en-v1.5'
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(os.path.join(BASE_DIR, "DAA-Collab"))
DATA_PATH = os.path.join(BASE_DIR, "DAA-Collab/data/user_prompt_dataset/traffic_prompts_realistic_option2_FIXED.csv")
MODEL_DIR = os.path.join(BASE_DIR,"DAA-Collab/utils/ML/ml_based_intent_classification/saved_models_transformer")

os.makedirs(MODEL_DIR, exist_ok=True)

# Load pre-trained sentence transformer (cache for offline use)
try:
    # model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODEL_DIR) 
    model = SentenceTransformer(MODEL_NAME, cache_folder = MODEL_DIR, trust_remote_code=True)
    print("✅ Sentence Transformer model loaded")
except Exception as e:
    print(f"⚠️ Error loading Sentence Transformer: {e}")
    print("   Run: pip install sentence-transformers")
    model = None


def train_intent_classifier(include_memory_data: bool = False):
    """
    Train semantic similarity based classifier.
    
    OFFLINE TRAINING ONLY - Should NOT be called during inference.
    
    Args:
        include_memory_data: Whether to include validated interactions from memory
    """
    if model is None:
        raise RuntimeError("Sentence Transformer not available. Install: pip install sentence-transformers")
    
    print("\n" + "="*80)
    print("TRAINING INTENT CLASSIFIER")
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
    intent_prototypes = {}
    intent_examples = {}
    
    # Group prompts by intent
    for _, row in df.iterrows():
        for intent in row['intents']:
            if intent not in intent_examples:
                intent_examples[intent] = []
            intent_examples[intent].append(row['prompt'])
    
    print(f"\nTotal unique intents: {len(intent_examples)}")
    print(f"Total training samples: {len(df)}")
    
    # Calculate train/test split (simulated for reporting)
    # Since this is a prototype-based few-shot learner, we use all data for "training" (building prototypes)
    # But for reporting purposes, we can show the breakdown.
    print(f"Training samples used for prototypes: {len(df)}")
    print(f"Test samples: 0 (All data used for prototype construction)")
    print("\nNOTE: This model uses Prototype Learning (Few-Shot).")
    print("      It does not require a traditional Train/Test split because it computes")
    print("      centroid embeddings for each class rather than optimizing weights via backprop.")
    print("      All available data is used to build the most robust prototypes possible.")

    print("\nIntent distribution:")
    for intent, examples in sorted(intent_examples.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {intent}: {len(examples)} examples")
    
    print("\nCreating semantic embeddings...")
    
    # Create prototype embeddings for each intent
    for intent, examples in intent_examples.items():
        # Use top 100 examples per intent (or all if less)
        sample_examples = examples[:100]
        print(f"  Encoding {intent}: {len(sample_examples)} examples...")
        embeddings = model.encode(sample_examples, show_progress_bar=False)
        # Average embedding as prototype
        intent_prototypes[intent] = np.mean(embeddings, axis=0)
    
    # Save prototypes and classes
    joblib.dump(intent_prototypes, os.path.join(MODEL_DIR, "intent_prototypes.joblib"))
    joblib.dump(list(intent_examples.keys()), os.path.join(MODEL_DIR, "intent_classes.joblib"))
    
    print(f"\n✅ Model trained and saved to: {MODEL_DIR}")
    print("="*80 + "\n")


def load_intent_classifier() -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Load the trained semantic classifier.
    
    Returns:
        Tuple of (intent_prototypes, intent_classes)
    
    Raises:
        FileNotFoundError: If model files don't exist
    """
    prototypes_path = os.path.join(MODEL_DIR, "intent_prototypes.joblib")
    classes_path = os.path.join(MODEL_DIR, "intent_classes.joblib")
    
    if not os.path.exists(prototypes_path) or not os.path.exists(classes_path):
        raise FileNotFoundError(
            f"Model files not found in {MODEL_DIR}\n"
            "Train the model first: python -m utils.ML.ml_based_intent_classification.model_transformer"
        )
    
    intent_prototypes = joblib.load(prototypes_path)
    intent_classes = joblib.load(classes_path)
    
    return intent_prototypes, intent_classes


def predict_intents_ml(
    prompt: str, 
    threshold: float = 0.7, 
    debug: bool = True,
    store_in_memory: bool = False
) -> Tuple[List[str], Dict[str, float]]:
    """
    Predict intents using semantic similarity.
    
    INFERENCE ONLY - No training happens here.
    
    Args:
        prompt: User input text
        threshold: Similarity threshold (0.0-1.0, default 0.45)
        debug: Print debug information
        store_in_memory: DEPRECATED - memory handled by LangGraph node
    
    Returns:
        Tuple of (predicted_intents, confidence_scores)
        - predicted_intents: List of intent names above threshold
        - confidence_scores: Dict mapping all intents to their similarity scores (0.0-1.0)
    """
    if model is None:
        raise RuntimeError("Sentence Transformer not loaded")
    
    # Load model
    intent_prototypes, intent_classes = load_intent_classifier()
    
    # Preprocess prompt
    preprocessed = preprocess_prompt(prompt, debug=debug)
    cleaned_prompt = preprocessed['cleaned']
    
    # Get prompt embedding
    prompt_embedding = model.encode([cleaned_prompt], show_progress_bar=False)[0]
    
    # Calculate similarity to each intent prototype
    similarities = {}
    for intent in intent_classes:
        prototype = intent_prototypes[intent]
        similarity = cosine_similarity(
            np.array([prompt_embedding]), 
            np.array([prototype])
        )[0][0]
        # Ensure float for JSON serialization
        similarities[intent] = float(similarity)
    
    if debug:
        print(f"\n{'='*80}")
        print("SEMANTIC SIMILARITY SCORES")
        print(f"{'='*80}")
        for intent, score in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
            status = "✓" if score > threshold else " "
            print(f"  [{status}] {intent:25s}: {score:.3f}")
        print(f"{'='*80}\n")
    
    # Apply threshold
    predicted = [intent for intent, score in similarities.items() if score > threshold]
    
    # Fallback: if no intents detected, return top intent if above minimum
    if not predicted and similarities:
        top_intent, top_score = max(similarities.items(), key=lambda x: x[1])
        if top_score > 0.3:  # Minimum fallback threshold
            predicted = [top_intent]
            if debug:
                print(f"⚠️  Fallback: Using top intent '{top_intent}' (score: {top_score:.3f})\n")
    
    if debug:
        print(f"Predicted intents (threshold={threshold}): {predicted}\n")
    
    return predicted, similarities


if __name__ == "__main__":
    print("\n🚀 Training Intent Classification Model\n")
    train_intent_classifier(include_memory_data=False)