from typing import Dict, List
from utils.ML.intent_extraction_classification.intent_mapping import INTENT_MAPPING
from utils.ML.intent_extraction_classification.ml_intent_classification import predict_intents_ml

def user_intent_classification(intent_extracted: List[str]) -> Dict[str, bool]:
    
    # Initialize all intents as False
    intent_classification_result = {intent_key: False for intent_key in INTENT_MAPPING.keys()}
    
    # Check each extracted keyword against all intent buckets
    for keyword in intent_extracted:
        for intent_key, keyword_list in INTENT_MAPPING.items():
            if keyword in keyword_list:
                intent_classification_result[intent_key] = True
    
    return intent_classification_result


def user_intent_classification_keyword_based(intent_extracted: List[str]) -> Dict[str, bool]:
    """
    Keyword-based intent classification (original approach).
    """
    intent_classification_result = {intent_key: False for intent_key in INTENT_MAPPING.keys()}
    
    for keyword in intent_extracted:
        for intent_key, keyword_list in INTENT_MAPPING.items():
            if keyword in keyword_list:
                intent_classification_result[intent_key] = True
    
    return intent_classification_result


def user_intent_classification_ml_based(prompt: str) -> Dict[str, bool]:
    """
    ML-based intent classification using trained SVM model.
    """
    predicted_intents = predict_intents_ml(prompt)
    
    # Convert list of intents to boolean dict
    intent_classification_result = {intent_key: False for intent_key in INTENT_MAPPING.keys()}
    
    for intent in predicted_intents:
        if intent in intent_classification_result:
            intent_classification_result[intent] = True
    
    return intent_classification_result


def user_intent_classification_hybrid(prompt: str, intent_extracted: List[str]) -> Dict[str, bool]:
    """
    Hybrid approach: Use ML model first, fallback to keyword matching.
    Combines both approaches for better accuracy.
    """
    # Start with ML prediction
    ml_result = user_intent_classification_ml_based(prompt)
    
    # Enhance with keyword-based classification
    keyword_result = user_intent_classification_keyword_based(intent_extracted)
    
    # Merge results (OR operation - if either detects intent, mark as True)
    combined_result = {
        intent_key: ml_result[intent_key] or keyword_result[intent_key]
        for intent_key in INTENT_MAPPING.keys()
    }
    
    return combined_result