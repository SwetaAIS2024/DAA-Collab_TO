# -------------------------------
# Custom MCP tool generator node
# -------------------------------

import json
from func.state.agent_state import AgentState

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