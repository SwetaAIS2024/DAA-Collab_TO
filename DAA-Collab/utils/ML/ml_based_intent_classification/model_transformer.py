import pandas as pd
from ast import literal_eval
import joblib
import os
from typing import List, Dict
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "DAA-Collab/data/user_prompt_dataset/traffic_prompts_realistic_option2_FIXED.csv"
MODEL_DIR = "DAA-Collab/utils/ML/ml_based_intent_classification/saved_models_transformer"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load pre-trained sentence transformer (download once, works offline after)
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Sentence Transformer model loaded")
except Exception as e:
    print(f"⚠️ Error loading Sentence Transformer: {e}")
    print("   Run: pip install sentence-transformers")
    model = None


def remove_negated_terms(text: str) -> str:
    """
    Remove terms that appear in negation context.
    e.g., "do not detect anomalies" -> "do not detect"
    """
    negation_patterns = [
        r'\b(do not|don\'t|does not|doesn\'t|did not|didn\'t)\s+(\w+\s+)?(\w+)',
        r'\b(not|no|never|without)\s+(\w+\s+)?(\w+)',
        r'\b(no)\s+(\w+)',
    ]
    
    cleaned_text = text.lower()
    
    for pattern in negation_patterns:
        matches = re.finditer(pattern, cleaned_text)
        for match in matches:
            full_match = match.group(0)
            negation_word = match.group(1)
            cleaned_text = cleaned_text.replace(full_match, negation_word)
    
    return cleaned_text


def train_intent_classifier():
    """Train semantic similarity based classifier"""
    if model is None:
        print("⚠️ Sentence Transformer not available. Cannot train.")
        return
    
    df = pd.read_csv(DATA_PATH)
    df["intents"] = df["intents"].apply(literal_eval)

    # Create intent prototypes (average embeddings per intent)
    intent_prototypes = {}
    intent_examples = {}
    
    # Group prompts by intent
    for _, row in df.iterrows():
        for intent in row['intents']:
            if intent not in intent_examples:
                intent_examples[intent] = []
            intent_examples[intent].append(row['prompt'])
    
    print(f"\nTotal unique intents: {len(intent_examples)}\n")
    print("Unique intent labels:")
    for intent in sorted(intent_examples.keys()):
        print(f"  - {intent}")
    
    print("\n\nIntent distribution:")
    for intent, examples in sorted(intent_examples.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {intent}: {len(examples)} examples")
    
    print("\n\nCreating semantic embeddings for intents...")
    
    # Create prototype embeddings for each intent
    for intent, examples in intent_examples.items():
        # Use top 100 examples per intent (or all if less)
        sample_examples = examples[:100]
        print(f"  Encoding {intent}: {len(sample_examples)} examples...")
        embeddings = model.encode(sample_examples, show_progress_bar=False)
        # Average embedding as prototype
        intent_prototypes[intent] = np.mean(embeddings, axis=0)
    
    # Save prototypes
    joblib.dump(intent_prototypes, os.path.join(MODEL_DIR, "intent_prototypes.joblib"))
    joblib.dump(list(intent_examples.keys()), os.path.join(MODEL_DIR, "intent_classes.joblib"))
    
    print("\n✅ Semantic intent classifier trained and saved.")
    print(f"   Model saved to: {MODEL_DIR}")


def load_intent_classifier():
    """Load the trained semantic classifier"""
    intent_prototypes = joblib.load(os.path.join(MODEL_DIR, "intent_prototypes.joblib"))
    intent_classes = joblib.load(os.path.join(MODEL_DIR, "intent_classes.joblib"))
    return intent_prototypes, intent_classes


def predict_intents_ml(prompt: str, threshold: float = 0.45, debug: bool = True) -> List[str]:
    """
    Predict intents using semantic similarity.
    
    Args:
        prompt: User input text
        threshold: Similarity threshold (0.0-1.0, default 0.45 for balanced detection)
        debug: Print debug information
    
    Returns:
        List of predicted intent labels
    """
    try:
        if model is None:
            print("⚠️ Sentence Transformer not available.")
            return []
        
        intent_prototypes, intent_classes = load_intent_classifier()
        
        # Preprocess prompt to remove negated terms
        cleaned_prompt = remove_negated_terms(prompt)
        
        if debug:
            print(f"\nOriginal prompt: {prompt}")
            print(f"Cleaned prompt: {cleaned_prompt}")
        
        # Get prompt embedding
        prompt_embedding = model.encode([cleaned_prompt], show_progress_bar=False)[0]
        
        # Calculate similarity to each intent prototype
        similarities = {}
        for intent in intent_classes:
            prototype = intent_prototypes[intent]
            similarity = cosine_similarity(np.array([prompt_embedding]), np.array([prototype]))[0][0]
            similarities[intent] = similarity
        
        if debug:
            print(f"\nSemantic similarity scores by intent:")
            for intent, score in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
                print(f"  {intent}: {score:.3f}")
        
        # Apply threshold
        predicted = [intent for intent, score in similarities.items() if score > threshold]
        
        if debug:
            print(f"\nPredicted intents (threshold={threshold}): {predicted}")
        
        # Fallback: if no intents detected, return top intent if above minimum threshold
        if not predicted and similarities:
            top_intent, top_score = max(similarities.items(), key=lambda x: x[1])
            if top_score > 0.3:  # Minimum threshold
                predicted = [top_intent]
                if debug:
                    print(f"Fallback: Using top intent {top_intent} (score: {top_score:.3f})")
        
        return predicted
        
    except FileNotFoundError:
        print("⚠️ ML models not found. Run train_intent_classifier() first.")
        return []
    except Exception as e:
        print(f"⚠️ ML prediction error: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    train_intent_classifier()