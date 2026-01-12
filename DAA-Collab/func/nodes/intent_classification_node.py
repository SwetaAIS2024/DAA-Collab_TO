from func.state.agent_state import AgentState
from utils.ML.ml_based_intent_classification.intent_classification import user_query_intent_extraction
from utils.common import preprocess_prompt
import uuid

def intent_classification(state: AgentState) -> AgentState:

    user_instruction = state.instruction
    
    # Generate execution ID if not present
    if not state.execution_id:
        state.execution_id = f"exec_{uuid.uuid4().hex[:12]}"
    
    print(f"Intent Classification Node - Processing: {user_instruction}")
    
    try:
        intent_threshold = 0.75  # Lower threshold, but always select at least 3
        min_intents = 3  # Always return at least 3 intents
        # Get preprocessed prompt for metadata
        preprocessed = preprocess_prompt(user_instruction, debug=False)

        # Perform intent classification
        intent_dict, confidence_scores, top_known_intents, unknown_intents = user_query_intent_extraction(
            prompt=preprocessed['cleaned'],
            threshold=intent_threshold,
            min_intents=min_intents,
            debug=True
        )
        
        print(f"Classified Intents: {intent_dict['intents']}")
        
        # Update state
        state.intent_classification = intent_dict
        state.predicted_classified_intents = intent_dict['intents']  # All intents
        state.known_intents = top_known_intents
        state.unknown_intents = unknown_intents
        state.intent_confidence_scores = confidence_scores
        state.classification_success = True
        state.needs_tool_generation = len(unknown_intents) > 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        state.classification_success = False
        state.error_log = [str(e)]
        
    except Exception as e:
        print(f"Error: {e}")
        state.classification_success = False
        state.error_log = [str(e)]
    
    return state