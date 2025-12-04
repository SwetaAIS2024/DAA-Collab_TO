from typing import Dict, List
from intent_mapping import INTENT_MAPPING


def user_intent_classification(intent_extracted: List[str]) -> Dict[str, bool]:
    
    # Initialize all intents as False
    intent_classification_result = {intent_key: False for intent_key in INTENT_MAPPING.keys()}
    
    # Check each extracted keyword against all intent buckets
    for keyword in intent_extracted:
        for intent_key, keyword_list in INTENT_MAPPING.items():
            if keyword in keyword_list:
                intent_classification_result[intent_key] = True
    
    return intent_classification_result