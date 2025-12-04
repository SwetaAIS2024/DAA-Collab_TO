# -------------------------------
# Custom intent classification node
# -------------------------------
from typing import Dict
from func.state.agent_state import AgentState
from utils.ML.intent_extraction_classification.intent_extraction import user_intent_extraction
from utils.ML.intent_extraction_classification.intent_classification import user_intent_classification

def classify_intent(instr: str) -> Dict[str, bool]:
    low = instr.lower()
    return {
        "visualization": any(k in low for k in ["plot", "chart", "graph", "histogram", "scatter", "box", "bar", "visualize", "pairplot"]),
        "advanced": any(k in low for k in ["eda", "analysis", "correlation", "cluster", "regression", "patterns", "pairplot"]),
    }


def intent_classifier_node(state: AgentState) -> AgentState:
    """Classify user intent from instruction."""

    user_instruction = state.instruction.lower()
    intent_extraction_result = user_intent_extraction(user_instruction)
    intent_classification_result = user_intent_classification(intent_extraction_result)
    state.intent_classification = intent_classification_result

    return state