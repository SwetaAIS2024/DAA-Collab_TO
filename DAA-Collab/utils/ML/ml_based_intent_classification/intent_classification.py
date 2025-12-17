from typing import Dict, List, Tuple
import os
from utils.ML.ml_based_intent_classification.model_transformer import (
    predict_intents_ml, 
    load_intent_classifier
)

def user_intent_classification_ml_based(
    prompt: str, 
    threshold: float = 0.7,
    debug: bool = True
) -> Tuple[Dict[str, List[str]], Dict[str, float], List[str], List[str]]:
    """
    ML-based intent classification using Sentence Transformers semantic similarity.
    
    PRODUCTION-READY: Does NOT train during inference.
    Model must be pre-trained before deployment.
    
    Args:
        prompt: User input text
        threshold: Similarity threshold (0.0-1.0, default 0.45)
    
    Returns:
        Tuple of (intent_dict, confidence_scores, predicted_intents)
        - intent_dict: Dict[intent_name, bool] for all intents
        - confidence_scores: Dict[intent_name, float] similarity scores
        - predicted_intents: List[str] of intents above threshold
    
    Raises:
        FileNotFoundError: If model files not found (need to train first)
    """
    try:
        # Load the trained model (will raise error if not trained)
        _, intent_classes = load_intent_classifier()
        
    except FileNotFoundError:
        raise FileNotFoundError(
            "Intent classification model not found. Please train the model first:\n"
            "  python -m utils.ML.ml_based_intent_classification.model_transformer"
        )
    
    # Get predicted intents and confidence scores
    predicted_intents, confidence_scores, unknown_intents = predict_intents_ml(
        prompt=prompt,
        threshold=threshold,
        debug=debug
    )
    
    all_intents = predicted_intents.copy()

    for unknown in unknown_intents:
        all_intents.append(unknown)

    # # Convert to boolean dict with ALL possible intents
    # intent_classification_result = {
    #     intent: (intent in predicted_intents) 
    #     for intent in intent_classes
    # }

    intent_classification_result = {
        "intents": all_intents,
        "known_intents": predicted_intents,
        "unknown_intents": unknown_intents
    }


    return intent_classification_result, confidence_scores, predicted_intents, unknown_intents