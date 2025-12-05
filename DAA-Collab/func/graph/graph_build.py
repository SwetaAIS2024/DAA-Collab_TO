# -------------------------------
# Graph construction
# -------------------------------

# LangGraph minimal imports
from langgraph.graph import StateGraph, END
from func.state.agent_state import AgentState
from func.nodes.planner_node import planner_node
from func.nodes.execute_node import execute_node
from func.nodes.intent_classifier_node import intent_classifier_node
from func.nodes.mcp_tool_generator_node import mcp_tool_generator_node

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