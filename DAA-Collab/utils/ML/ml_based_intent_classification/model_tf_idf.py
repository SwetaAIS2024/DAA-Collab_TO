import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import joblib
import os
from typing import List, Tuple
import re
import numpy as np


DATA_PATH = "DAA-Collab/data/user_prompt_dataset/traffic_prompts_realistic_option2_FIXED.csv"
MODEL_DIR = "DAA-Collab/utils/ML/ml_based_intent_classification/saved_models_tf_tdf"
os.makedirs(MODEL_DIR, exist_ok=True)


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
    """Train and save the ML intent classifier"""
    df = pd.read_csv(DATA_PATH)
    
    # Parse the intents column
    df["intents"] = df["intents"].apply(literal_eval)

    # Extract all unique intents
    all_intents = set()
    for intent_list in df['intents']:
        all_intents.update(intent_list)

    unique_intents = sorted(all_intents)

    print(f"Total unique intents: {len(unique_intents)}\n")
    print("Unique intent labels:")
    for intent in unique_intents:
        print(f"  - {intent}")

    # Intent distribution
    intent_counts = {}
    for intent_list in df['intents']:
        for intent in intent_list:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

    print("\n\nIntent distribution:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {intent}: {count}")

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["intents"])

    # Improved TF-IDF settings for better feature extraction
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),  # Include trigrams for phrases like "spatial trend"
        min_df=1,            # Lower threshold to catch rare terms
        max_df=0.95,         # Slightly higher to keep more features
        stop_words="english",
        sublinear_tf=True    # Use log scaling
    )

    X = vectorizer.fit_transform(df["prompt"])

    # Use probability calibrated classifier for better multi-label prediction
    classifier = OneVsRestClassifier(
        LinearSVC(class_weight="balanced", C=1.0, max_iter=2000)
    )
    classifier.fit(X, Y)

    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "intent_vectorizer.joblib"))
    joblib.dump(classifier, os.path.join(MODEL_DIR, "intent_classifier.joblib"))
    joblib.dump(mlb, os.path.join(MODEL_DIR, "intent_mlb.joblib"))

    print("\n✅ Intent classifier trained and saved.")


def load_intent_classifier():
    """Load the trained ML classifier"""
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "intent_vectorizer.joblib"))
    classifier = joblib.load(os.path.join(MODEL_DIR, "intent_classifier.joblib"))
    mlb = joblib.load(os.path.join(MODEL_DIR, "intent_mlb.joblib"))
    return vectorizer, classifier, mlb


def predict_intents_ml(prompt: str, threshold: float = -0.5, debug: bool = True) -> List[str]:
    """
    Predict intents using the trained ML model.
    
    Args:
        prompt: User input text
        threshold: Decision threshold (lower = more permissive, default -0.5 to catch more intents)
        debug: Print debug information
    
    Returns:
        List of predicted intent labels
    """
    try:
        vectorizer, classifier, mlb = load_intent_classifier()
        
        # Preprocess prompt to remove negated terms
        cleaned_prompt = remove_negated_terms(prompt)
        
        if debug:
            print(f"\nOriginal prompt: {prompt}")
            print(f"Cleaned prompt: {cleaned_prompt}")
        
        X = vectorizer.transform([cleaned_prompt])

        # Get decision scores for all classes
        decision_scores = classifier.decision_function(X)[0]
        
        if debug:
            print(f"\nDecision scores by intent:")
            for intent, score in sorted(zip(mlb.classes_, decision_scores), key=lambda x: x[1], reverse=True):
                print(f"  {intent}: {score:.3f}")
        
        # Apply threshold
        y_pred = (decision_scores > threshold).astype(int).reshape(1, -1)
        
        if debug:
            print(f"\nPrediction vector (threshold={threshold}): {y_pred}")

        intents = mlb.inverse_transform(y_pred)
        predicted = list(intents[0]) if intents and len(intents[0]) > 0 else []
        
        if debug:
            print(f"Predicted intents: {predicted}")
        
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