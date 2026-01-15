"""
Test intent classification using remote inference server
Runs full intent classification logic on client, uses server only for model inference

Usage:
    python test_remote_intent.py "Your query here"
    python test_remote_intent.py "Your query here" --server http://127.0.0.1:8000
"""

import sys
import os
from pathlib import Path
import argparse
import numpy as np
from typing import List, Dict, Tuple

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference_server.client import InferenceClient


def load_intent_centroids():
    """Load saved intent centroids from local files"""
    import joblib
    
    model_dir = Path(__file__).parent.parent / "utils" / "ML" / "ml_based_intent_classification" / "saved_models_transformer"
    
    prototypes_path = model_dir / "intent_centroids.joblib"
    classes_path = model_dir / "intent_classes.joblib"
    
    if not prototypes_path.exists() or not classes_path.exists():
        raise FileNotFoundError(
            f"Model files not found in {model_dir}\n"
            "Make sure intent_centroids.joblib and intent_classes.joblib exist"
        )
    
    intent_prototypes = joblib.load(prototypes_path)
    intent_classes = joblib.load(classes_path)
    
    return intent_prototypes, intent_classes


def classify_intent_remote(
    query: str,
    client: InferenceClient,
    threshold: float = 0.6,
    min_intents: int = 1,
    debug: bool = True
) -> Tuple[List[str], Dict[str, float], List[str], List[str]]:
    """
    Classify intent using remote server for inference
    All logic runs locally, only model inference is remote
    """
    
    # Load intent classes (local)
    intent_prototypes, intent_classes = load_intent_centroids()
    
    # Get embeddings from remote server
    if debug:
        print("Getting embeddings from remote server...")
    
    # Encode query and all intent classes
    all_texts = [query] + intent_classes
    embeddings = client.get_embeddings(all_texts)
    
    query_embedding = embeddings[0]
    intent_embeddings = embeddings[1:]
    
    # Compute similarities (local - cheap CPU operation)
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarities = {}
    for i, intent in enumerate(intent_classes):
        sim = cosine_similarity(
            query_embedding.reshape(1, -1),
            intent_embeddings[i].reshape(1, -1)
        )[0][0]
        similarities[intent] = float(sim)
    
    # Get top intent and score
    top_intent, top_score = max(similarities.items(), key=lambda x: x[1])
    sorted_intents = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # Select predicted intents based on threshold
    predicted = [intent for intent, score in sorted_intents if score >= threshold]
    
    # Ensure at least min_intents
    if len(predicted) < min_intents:
        predicted = [intent for intent, _ in sorted_intents[:min_intents]]
    
    # Check if we should extract unknown intents
    extract_unknown = top_score < 0.85
    
    if extract_unknown:
        if debug:
            print("Extracting unknown intents from remote LLM...")
        
        unknown_intents = client.extract_unknown_intents(
            query=query,
            known_intents=predicted,
            known_classes=intent_classes
        )
    else:
        unknown_intents = []
    
    # Combine all intents
    all_intents = predicted + unknown_intents
    
    return all_intents, similarities, predicted, unknown_intents


def test_query(query: str, server_url: str):
    """Test intent classification for a query using remote server"""
    
    print(f"\nQuery: {query}")
    print("-" * 60)
    
    # Initialize remote client
    try:
        client = InferenceClient(server_url)
        print()
    except Exception as e:
        print(f"Error connecting to server: {e}")
        print("\nMake sure:")
        print("1. Server is running on Windows")
        print("2. SSH tunnel is active: ssh -L 8000:127.0.0.1:8000 admin@<windows-ip>")
        return False
    
    # Run classification
    try:
        all_intents, similarities, known_intents, unknown_intents = classify_intent_remote(
            query=query,
            client=client,
            threshold=0.6,
            min_intents=1,
            debug=True
        )
        
        print(f"\n{'='*20} RESULTS {'='*20}\n")
        print(f"Classified Intents ({len(all_intents)}):")
        for intent in all_intents:
            score = similarities.get(intent, 0.0)
            print(f"  - {intent}: {score:.4f}")
        
        print(f"\nKnown Intents ({len(known_intents)}): {known_intents}")
        print(f"Unknown Intents ({len(unknown_intents)}): {unknown_intents}")
        
        return True
        
    except Exception as e:
        print(f"\nClassification error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test intent classification with remote server")
    parser.add_argument("query", nargs="+", help="Query text to classify")
    parser.add_argument("--server", default="http://127.0.0.1:8000", help="Server URL")
    
    args = parser.parse_args()
    query = " ".join(args.query)
    
    success = test_query(query, args.server)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
