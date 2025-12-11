# -------------------------------
# Custom intent classification node
# -------------------------------
from typing import Dict
from func.state.agent_state import AgentState

from utils.ML.ml_based_intent_classification.intent_classification import user_intent_classification_ml_based



def intent_classification(state: AgentState) -> AgentState:
    """
    LangGraph node: Extract and classify user intent using hybrid approach.
    Combines ML model + keyword matching for robust intent detection.
    """
    user_instruction = state.instruction
    intent_classification_result = user_intent_classification_ml_based(prompt=user_instruction)
    state.intent_classification = intent_classification_result
    
    return state