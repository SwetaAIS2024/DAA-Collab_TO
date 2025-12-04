# -------------------------------
# Execute node
# -------------------------------

from func.state.agent_state import AgentState
from typing import List
# from utils.mcp_tools_registry. import TOOL_MAP

"""Tool functions are now sourced from external mcp_tools.py."""
from utils.mcp_tools_registry.static_tools.mcp_tools import (
    load_and_analyze_csv,
    perform_advanced_eda_on_csv,
    generate_basic_plot,
)

TOOL_MAP = {
    "load_and_analyze_csv": load_and_analyze_csv,
    "perform_advanced_eda_on_csv": perform_advanced_eda_on_csv,
    "generate_basic_plot": generate_basic_plot,
}

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