# -------------------------------
# State definition for Intent Classification Node
# -------------------------------

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class AgentState:
    # Input
    instruction: Optional[str] = None
    
    # Error handling
    error_log: List[str] = field(default_factory=list)

    # Intent classification outputs
    intent_classification: Optional[Dict[str, List[str]]] = None
    predicted_classified_intents: List[str] = field(default_factory=list)
    intent_confidence_scores: Dict[str, float] = field(default_factory=dict)
    classification_success: bool = False
    known_intents: List[str] = field(default_factory=list)
    unknown_intents: List[str] = field(default_factory=list)
    needs_tool_generation: bool = False

    # Execution tracking
    execution_id: Optional[str] = None


