# # -------------------------------
# # Custom intent classification node
# # -------------------------------
# from typing import Dict
# from func.state.agent_state import AgentState

# from utils.ML.ml_based_intent_classification.intent_classification import user_intent_classification_ml_based



# def intent_classification(state: AgentState) -> AgentState:
#     """
#     LangGraph node: Extract and classify user intent using hybrid approach.
#     Combines ML model + keyword matching for robust intent detection.
#     """
#     user_instruction = state.instruction
#     intent_classification_result = user_intent_classification_ml_based(prompt=user_instruction)
#     state.intent_classification = intent_classification_result
    
#     return state



from func.state.agent_state import AgentState
from utils.ML.ml_based_intent_classification.intent_classification import user_intent_classification_ml_based
from utils.memory.episodic import get_intent_memory, store_intent_interaction
from utils.common import preprocess_prompt
from datetime import datetime
import uuid

def intent_classification(state: AgentState) -> AgentState:
    """
    Intent Classification Node with LangGraph Memory Integration.
    
    PRODUCTION-READY:
    - Uses pre-trained model (no training during inference)
    - Consistent confidence score handling
    - Automatic memory tracking in LangGraph
    - Full error handling
    """
    user_instruction = state.instruction
    
    # Generate execution ID if not present
    if not state.execution_id:
        state.execution_id = f"exec_{uuid.uuid4().hex[:12]}"
    
    print("\n" + "="*80)
    print("🎯 INTENT CLASSIFICATION NODE")
    print("="*80)
    print(f"Execution ID: {state.execution_id}")
    print(f"User Input: {user_instruction}\n")
    
    try:
        # Perform intent classification
        intent_dict, confidence_scores, predicted_intents = user_intent_classification_ml_based(
            prompt=user_instruction,
            threshold=0.85
        )
        
        # Get preprocessed prompt for metadata
        preprocessed = preprocess_prompt(user_instruction, debug=False)
        
        # Store in LangGraph memory
        memory = get_intent_memory()
        checkpoint_id = memory.store_interaction(
            prompt=user_instruction,
            cleaned_prompt=preprocessed['cleaned'],
            predicted_intents=predicted_intents,
            confidence_scores=confidence_scores,  # Now consistent Dict[str, float]
            metadata={
                'temporal_entities': preprocessed.get('temporal_entities', []),
                'spatial_entities': preprocessed.get('spatial_entities', []),
                'threshold': 0.85,
                'node_name': 'intent_classification',
                'graph_execution_id': state.execution_id,
                'preprocessing_steps': preprocessed.get('processing_steps', [])
            }
        )
        
        print(f"💾 Stored in LangGraph Memory")
        print(f"   Checkpoint ID: {checkpoint_id}")
        print(f"   Detected Intents: {predicted_intents}")
        print(f"\nTop 5 Confidence Scores:")
        for intent, score in sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
            status = "✓" if intent in predicted_intents else " "
            print(f"   [{status}] {intent:25s}: {score:.3f}")
        
        # Update state
        state.intent_classification = intent_dict
        state.mem_checkpoint_id = checkpoint_id
        state.predicted_classified_intents = predicted_intents
        state.intent_confidence_scores = confidence_scores
        state.classification_success = True
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("   Please train the model before running inference.")
        state.classification_success = False
        state.error_log = [str(e)]
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        state.classification_success = False
        state.error_log = [str(e)]
    
    print("="*80 + "\n")
    
    return state