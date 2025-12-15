# -------------------------------
# State definition
# -------------------------------

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class AgentState:
    instruction: str
    dataset_path: str
    seed_text: Optional[str] = None
    profile_report: Optional[str] = None
    profile_meta: Optional[Dict[str, Any]] = None
    plan_json: Optional[str] = None
    plan_steps: List[Dict[str, Any]] = field(default_factory=list)
    next_step_index: int = 0
    raw_execution_output: Optional[str] = None
    error_log: List[str] = field(default_factory=list)

    # ML-based intent classification results
    intent_classification: Optional[Dict[str, List[str]]] = None
    predicted_classified_intents: List[str] = field(default_factory=list)
    intent_confidence_scores: Dict[str, float] = field(default_factory=dict)
    classification_success: bool = False
    known_intents: List[str] = field(default_factory=list)
    unknown_intents: List[str] = field(default_factory=list)
    needs_tool_generation: bool = False

    # Graph execution state
    execution_id: Optional[str] = None # Unique ID for the graph execution
    mem_checkpoint_id : Optional[str] = None # Memory checkpoint ID after last step

    # Feedback and validation
    feedback_collected : bool = False
    validated_intents : List[str] = field(default_factory=list)

    # Tool usage tracking
    missing_tools: List[str] = field(default_factory=list)


