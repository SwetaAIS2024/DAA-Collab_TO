# -------------------------------
# CLI / Runner
# -------------------------------

from func.graph.graph_build_complex import build_graph
from func.state.agent_state import AgentState
from utils.common import make_seed_text

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

