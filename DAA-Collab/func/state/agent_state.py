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
    intent_classification: Optional[Dict[str, bool]] = None
    missing_tools: List[str] = field(default_factory=list)


