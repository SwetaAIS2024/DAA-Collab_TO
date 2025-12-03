"""Minimal LangGraph-based CSV analysis agent.

Flow: User supplies --instruction and --file (CSV).
1. Planner node builds JSON plan of tool steps.
2. Execute node runs each step sequentially and prints raw concatenated output.

Single-file implementation.
"""
from __future__ import annotations
import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

from pathlib import Path

# LangGraph minimal imports
from langgraph.graph import StateGraph, END

from utils.intent_extraction import user_intent_extraction
from utils.intent_classification import user_intent_classification
from utils.smart_mcp_tool_lookup import ToolMap

"""Tool functions are now sourced from external mcp_tools.py."""
import mcp_tools
import inspect

# Auto-discover only MCP tool functions (decorated with @mcp.tool())
legacy_tool_dictionary = {}
for name, obj in inspect.getmembers(mcp_tools, inspect.isfunction):
    # Only include functions actually defined in mcp_tools (exclude imports)
    if not name.startswith('_') and hasattr(obj, '__module__') and obj.__module__ == 'mcp_tools':
        # Optional: Further filter to only @mcp.tool() decorated functions
        # Check if function has MCP tool markers (decorated functions often have special attributes)
        # For now, we include all functions - helper functions won't break anything
        legacy_tool_dictionary[name] = obj

TOOL_MAP = ToolMap(legacy_tool_dictionary)

# -------------------------------
# State definition
# -------------------------------
@dataclass
class AgentState:
    instruction: str
    dataset_path: str
    seed_text: str
    profile_report: Optional[str] = None
    profile_meta: Optional[Dict[str, Any]] = None
    plan_json: Optional[str] = None
    plan_steps: List[Dict[str, Any]] = field(default_factory=list)
    next_step_index: int = 0
    raw_execution_output: Optional[str] = None
    error_log: List[str] = field(default_factory=list)
    intent_classification: Optional[Dict[str, bool]] = None

# -------------------------------
# Utility helpers
# -------------------------------
FOOTER_PATTERN = r"<!--output_json:(.+?)-->"


def make_seed_text(instruction: str, dataset_path: str) -> str:
    return f"Dataset: {dataset_path or 'unspecified'}\nInstructions: {instruction}"


def parse_footer(md: str) -> Dict[str, Any]:
    m = re.search(FOOTER_PATTERN, md, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}


def classify_intent(instr: str) -> Dict[str, bool]:
    low = instr.lower()
    return {
        "visualization": any(k in low for k in ["plot", "chart", "graph", "histogram", "scatter", "box", "bar", "visualize", "pairplot"]),
        "advanced": any(k in low for k in ["eda", "analysis", "correlation", "cluster", "regression", "patterns", "pairplot"]),
    }


def classify_intent_custom(instr: str) -> Dict[str, bool]:

    user_instruction = instr.lower()

    intent_extraction_result = user_intent_extraction(user_instruction)

    intent_classification_result = user_intent_classification(intent_extraction_result)

    return intent_classification_result


def infer_plot_type(instr: str) -> str:
    low = instr.lower()
    if "histogram" in low: return "histogram"
    if "scatter" in low: return "scatterplot"
    if "box" in low: return "boxplot"
    if "pairplot" in low: return "pairplot"
    if "bar" in low or "chart" in low: return "bar_chart"
    # fallback
    return "histogram"


def infer_plot_columns(plot_type: str, numeric: List[str], categorical: List[str]) -> Optional[Dict[str, Any]]:
    if plot_type == "histogram" and numeric:
        return {"x_column": numeric[0]}
    if plot_type == "scatterplot" and len(numeric) >= 2:
        return {"x_column": numeric[0], "y_column": numeric[1]}
    if plot_type == "boxplot" and numeric and categorical:
        return {"x_column": numeric[0], "hue_column": categorical[0]}
    if plot_type == "bar_chart" and categorical:
        return {"x_column": categorical[0]}
    if plot_type == "pairplot" and len(numeric) >= 2:
        return {"columns_for_pairplot": numeric[:5]}
    return None



# -------------------------------
# Planner node
# -------------------------------

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

# -------------------------------
# Execute node
# -------------------------------


def execute_node(state: AgentState) -> AgentState:
    outputs: List[str] = []
    for step in state.plan_steps:
        tool_name = step.get("tool")
        fn = TOOL_MAP.get(tool_name)
        if not fn:
            outputs.append(f"Skipped unknown tool: {tool_name}")
            continue
        try:
            out = fn(**step.get("args", {}))
        except Exception as e:
            err = f"Error executing {tool_name}: {e}"
            state.error_log.append(err)
            out = err
        outputs.append(out)
    state.raw_execution_output = "\n\n--- STEP END ---\n\n".join(outputs)
    return state

# -------------------------------
# Custom intent classification node
# -------------------------------

def intent_classifier_node(state: AgentState) -> AgentState:
    """Classify user intent from instruction."""
    state.intent_classification = classify_intent_custom(state.instruction)
    return state

# -------------------------------
# Custom MCP tool generator node
# -------------------------------

def mcp_tool_generator_node(state: AgentState) -> AgentState:
    """Placeholder for MCP tool generator node."""
    # Currently, this node does not modify the state.
    if state.plan_steps:
        # If plan_steps already exist, skip tool generation
        return state

    else :

                # Plan is empty, ask LLM to create full plan
        prompt = f"""
                    Given:
                    - User instruction: {state.instruction}
                    - Detected intents: {state.intent_classification}
                    - Dataset info: {state.profile_meta}

                    Generate a JSON plan with steps to execute.
                    Each step should have: tool, args, why
                """
                        
        # TODO: Call on-prem LLM
        # steps_llm = call_onprem_llm(prompt)
        # state.plan_json = json.dumps({"steps": steps_llm}, ensure_ascii=False)
        # state.plan_steps = steps_llm
        # state.next_step_index = 0
        
        state.error_log.append("TODO: Generate plan with LLM")
        return state

# -------------------------------
# Graph construction
# -------------------------------

def build_graph():

    # Create graph
    g = StateGraph(AgentState)
    
    # Add nodes
    g.add_node("intent_classifier", intent_classifier_node)
    g.add_node("planner", planner_node)
    # placeholder for mcp tool generator
    g.add_node("tool_generator", mcp_tool_generator_node)
    g.add_node("executor", execute_node)
    
    # Define edges
    # g.set_entry_point("planner")
    g.set_entry_point("intent_classifier")
    g.add_edge("intent_classifier", "planner")
    # g.add_edge("planner", "executor")
    g.add_edge("planner", "tool_generator")
    g.add_edge("tool_generator", "executor")
    g.add_edge("executor", END)
    
    # Compile graph
    return g.compile()

# -------------------------------
# CLI / Runner
# -------------------------------

def run(instruction: str, dataset_path: str):
    state = AgentState(
        instruction=instruction,
        dataset_path=dataset_path,
        seed_text=make_seed_text(instruction, dataset_path),
    )
    app = build_graph()
    final_state = app.invoke(state)
    # Support both AgentState return and mapping-like return (e.g., AddableValuesDict)
    try:
        plan_json = final_state.plan_json
        raw_output = final_state.raw_execution_output
        errors = final_state.error_log
    except Exception:
        # final_state may be mapping-like; try .get
        try:
            plan_json = final_state.get("plan_json")
            raw_output = final_state.get("raw_execution_output")
            errors = final_state.get("error_log")
        except Exception:
            plan_json = None
            raw_output = None
            errors = None

    print("==== PLAN JSON ====\n" + (plan_json or "{}"))
    print("\n==== RAW EXECUTION OUTPUT ====\n" + (raw_output or ""))
    if errors:
        print("\nErrors:")
        for e in errors:
            print("-", e)


def main():
    parser = argparse.ArgumentParser(description="Minimal CSV analysis agent")
    parser.add_argument("--instruction", required=True, help="User instruction/query")
    parser.add_argument("--file", required=True, help="Path to CSV dataset")
    args = parser.parse_args()
    run(args.instruction, args.file)

if __name__ == "__main__":
    main()
