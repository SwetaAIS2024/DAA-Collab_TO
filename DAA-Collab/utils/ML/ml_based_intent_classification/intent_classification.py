from typing import Dict, List
# from utils.ML.ml_based_intent_classification.model_tf_idf import predict_intents_ml, load_intent_classifier, train_intent_classifier
from utils.ML.ml_based_intent_classification.model_transformer import predict_intents_ml, load_intent_classifier, train_intent_classifier

def user_intent_classification_ml_based(prompt: str, threshold: float = 0.5001) -> Dict[str, bool]:
    """
    ML-based intent classification using trained SVM model.
    Returns dict of all possible intents with True/False values.
    
    Args:
        prompt: User input text
        threshold: Decision threshold (default -0.5 for better multi-label detection)
    """
    # Train the model if not already trained
    train_intent_classifier()
    
    # Load the trained model to get all possible intent classes
    # _, _, mlb = load_intent_classifier() for TF-IDF model
    _, mlb = load_intent_classifier()  # for Transformer model
    
    # Get predicted intents from ML model with adjusted threshold
    predicted_intents = predict_intents_ml(prompt, threshold=threshold, debug=True)
    
    # Convert to boolean dict with ALL possible intents
    intent_classification_result = {
        intent: (intent in predicted_intents) 
        # for intent in mlb.classes_ # for TF-IDF model
        for intent in mlb  # for Transformer model
    }
    
    return intent_classification_result