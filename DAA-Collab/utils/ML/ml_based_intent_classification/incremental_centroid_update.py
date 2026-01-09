"""
Incremental centroid update - Add new intents without full rebuild
"""

import joblib
import os
import numpy as np
from typing import List
import threading

# Thread lock to prevent concurrent updates
_update_lock = threading.Lock()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
MODEL_DIR = os.path.join(BASE_DIR, "DAA-Collab/utils/ML/ml_based_intent_classification/saved_models_transformer")


def add_intent_to_centroids(intent_name: str, example_query: str) -> bool:
    """
    Add a single new intent to centroids without full rebuild
    
    Args:
        intent_name: Name of the new intent (e.g., "email_notification")
        example_query: The query that triggered this intent (used to compute centroid)
    
    Returns:
        True if successfully added, False otherwise
    """
    
    with _update_lock:
        try:
            # Paths
            centroids_path = os.path.join(MODEL_DIR, "intent_centroids.joblib")
            classes_path = os.path.join(MODEL_DIR, "intent_classes.joblib")
            
            # Load existing centroids
            if not os.path.exists(centroids_path) or not os.path.exists(classes_path):
                print(f"⚠️  Centroid files not found. Run build_centroids_script.py first.")
                return False
            
            centroids = joblib.load(centroids_path)
            classes = joblib.load(classes_path)
            
            # Check if intent already exists
            if intent_name in classes:
                print(f"ℹ️  Intent '{intent_name}' already exists in centroids. Skipping.")
                return True
            
            # Load embedding model
            from utils.ML.ml_based_intent_classification.intent_classification import get_embedding_model
            model = get_embedding_model()
            
            # Compute embedding for the example query
            embedding = model.encode([example_query], show_progress_bar=False)[0]
            
            # Add to centroids and classes
            centroids[intent_name] = embedding
            classes.append(intent_name)
            
            # Save updated centroids
            joblib.dump(centroids, centroids_path)
            joblib.dump(classes, classes_path)
            
            print(f"✅ Added '{intent_name}' to centroids (total: {len(classes)} intents)")
            return True
            
        except Exception as e:
            print(f"❌ Error adding intent to centroids: {e}")
            return False


def add_multiple_intents_to_centroids(intent_queries: List[tuple]) -> bool:
    """
    Add multiple new intents at once
    
    Args:
        intent_queries: List of (intent_name, example_query) tuples
    
    Returns:
        True if all successfully added, False otherwise
    """
    
    success_count = 0
    for intent_name, example_query in intent_queries:
        if add_intent_to_centroids(intent_name, example_query):
            success_count += 1
    
    return success_count == len(intent_queries)
