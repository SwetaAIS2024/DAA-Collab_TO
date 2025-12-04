# -------------------------------
# Planner node
# -------------------------------

import json
from pathlib import Path
from typing import Any, Dict, List
from func.state.agent_state import AgentState
from utils.common import parse_footer, infer_plot_type, infer_plot_columns
"""Tool functions are now sourced from external mcp_tools.py."""
from utils.mcp_tools_registry.static_tools.mcp_tools import load_and_analyze_csv


def planner_node(state: AgentState) -> AgentState:
    # Resolve dataset path relative to current working directory when needed
    ds_path = Path(state.dataset_path or "")
    if not ds_path.is_absolute():
        ds_path = Path.cwd() / ds_path
    ds_path = ds_path.resolve()
    if not ds_path.exists():
        state.plan_json = json.dumps({"steps": []})
        state.plan_steps = []
        state.error_log.append(f"Dataset file not found: {state.dataset_path}")
        return state
    # normalize to absolute path for downstream tools
    state.dataset_path = str(ds_path)

    if state.profile_meta is None:
        rep = load_and_analyze_csv(state.dataset_path)
        state.profile_report = rep
        state.profile_meta = parse_footer(rep)

    meta = state.profile_meta or {}
    numeric = meta.get("numeric_columns", [])
    categorical = meta.get("text_columns", [])
    
    # intent = classify_intent(state.instruction)
    
    intent = state.intent_classification or {}


    steps: List[Dict[str, Any]] = []
    # Always first overview
    steps.append({
        "tool": "load_and_analyze_csv",
        "args": {"file_path": state.dataset_path},
        "why": "dataset overview"
    })

    if intent["advanced"] and len(numeric) >= 2:
        steps.append({
            "tool": "perform_advanced_eda_on_csv",
            "args": {"file_path": state.dataset_path, "auto_detect_analysis": True},
            "why": "advanced EDA"
        })

    elif intent["visualization"]:
        plot_type = infer_plot_type(state.instruction)
        col_args = infer_plot_columns(plot_type, numeric, categorical)
        if col_args:
            args = {"file_path": state.dataset_path, "plot_type": plot_type, **col_args}
            steps.append({
                "tool": "generate_basic_plot",
                "args": args,
                "why": "requested visualization"
            })
    
    state.plan_json = json.dumps({"steps": steps}, ensure_ascii=False)
    state.plan_steps = steps
    state.next_step_index = 0
    return state