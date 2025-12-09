# ---------------------------------------------
# PLANNER SUBGRAPH
# ---------------------------------------------
from langgraph.graph import StateGraph, END
from func.state.agent_state import AgentState
from func.nodes.planner_sg_nodes.planner_node import plan_creation_node, plan_validation_node
from func.graph.subgraphs.replanning_sg import build_replanning_subgraph



def build_planner_subgraph():
    planner_subgraph = StateGraph(AgentState)

    # Internal planner nodes
    planner_subgraph.add_node("plan_creation", plan_creation_node)
    planner_subgraph.add_node("plan_validation", plan_validation_node)

    # Nested replanning subgraph
    planner_subgraph.add_node("replanning_subgraph", build_replanning_subgraph())

    planner_subgraph.set_entry_point("plan_creation")

    # plan_creation → plan_validation
    planner_subgraph.add_edge("plan_creation", "plan_validation")

    # Conditional validation → success OR → replanning
    planner_subgraph.add_conditional_edges(
        "plan_validation",
        validation_result,
        {
            True: END,
            False: "replanning_subgraph"
        }
    )

        # After replanning → go back to plan_creation
    planner_subgraph.add_edge("replanning_subgraph", "plan_creation")

    return planner_subgraph.compile()


def validation_result(state: AgentState) -> bool:
    # Placeholder logic for validation result
    return getattr(state, 'validation_passed', False)



# Build and compile the graph
graph = build_planner_subgraph()

# Generate Mermaid diagram
try:
    # Get Mermaid syntax
    mermaid_png = graph.get_graph().draw_mermaid_png()
    
    # Save as PNG
    with open("build_planner_subgraph_visualization.png", "wb") as f:
        f.write(mermaid_png)
    
    print("Graph saved as 'build_planner_subgraph_visualization.png'")
except Exception as e:
    print(f"Error generating PNG: {e}")
    # Fallback: print Mermaid syntax
    print("\nMermaid diagram syntax:")
    print(graph.get_graph().draw_mermaid())