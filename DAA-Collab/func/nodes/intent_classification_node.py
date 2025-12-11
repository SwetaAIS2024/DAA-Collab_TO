# -------------------------------
# Custom intent classification node
# -------------------------------
from typing import Dict
from func.state.agent_state import AgentState
from utils.ML.intent_extraction_classification.intent_extraction import user_intent_extraction
from utils.ML.intent_extraction_classification.intent_classification import user_intent_classification, user_intent_classification_hybrid


# def classify_intent(instr: str) -> Dict[str, bool]:
#     low = instr.lower()
#     return {
#         "visualization": any(k in low for k in ["plot", "chart", "graph", "histogram", "scatter", "box", "bar", "visualize", "pairplot"]),
#         "advanced": any(k in low for k in ["eda", "analysis", "correlation", "cluster", "regression", "patterns", "pairplot"]),
#     }


# def intent_classification(state: AgentState) -> AgentState:
#     """Classify user intent from instruction."""

#     user_instruction = state.instruction.lower()
#     intent_extraction_result = user_intent_extraction(user_instruction)
#     intent_classification_result = user_intent_classification(intent_extraction_result)
#     state.intent_classification = intent_classification_result

#     return state

def intent_classification(state: AgentState) -> AgentState:
    """
    LangGraph node: Extract and classify user intent using hybrid approach.
    Combines ML model + keyword matching for robust intent detection.
    """
    user_instruction = state.instruction
    
    # Step 1: Extract keywords from instruction
    intent_extracted = user_intent_extraction(user_instruction)
    
    # Step 2: Classify using hybrid approach (ML + Keywords)
    intent_classification_result = user_intent_classification_hybrid(
        prompt=user_instruction,
        intent_extracted=intent_extracted
    )
    
    # Update state
    state.intent_extracted = intent_extracted
    state.intent_classification = intent_classification_result
    
    return state