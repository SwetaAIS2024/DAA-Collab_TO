from langgraph.graph import StateGraph, END
from func.state.agent_state import AgentState
from func.graph.subgraphs.tool_creation_sg import build_tool_creation_subgraph
from func.nodes.replanning_sg_nodes.replanning_node import (
    tool_dependency_check_node,
    tool_mapping_node
)

def build_replanning_subgraph():
    replanning_subgraph = StateGraph(AgentState)

    replanning_subgraph.add_node("tool_dependency_check", tool_dependency_check_node)
    replanning_subgraph.add_node("tool_mapping", tool_mapping_node)

    # Nested tool creation pipeline
    replanning_subgraph.add_node("tool_creation_subgraph", build_tool_creation_subgraph())

    replanning_subgraph.set_entry_point("tool_dependency_check")

    # Conditional: tools missing?
    replanning_subgraph.add_conditional_edges(
        "tool_dependency_check",
        tools_missing,
        {
            True: "tool_creation_subgraph",
            False: "tool_mapping"
        }
    )

    # After tool creation → go to tool mapping
    replanning_subgraph.add_edge("tool_creation_subgraph", "tool_mapping")

    # After tool mapping → EXIT subgraph (return to parent)
    replanning_subgraph.add_edge("tool_mapping", END)

    return replanning_subgraph.compile()




# helper functions for the replanning subgraph
def tools_missing(state: AgentState) -> bool:
    # Placeholder logic to determine if tools are missing
    missing_tools = len(state.missing_tools) > 0
    return missing_tools



# Build and compile the graph
graph = build_replanning_subgraph()

# Generate Mermaid diagram
try:
    # Get Mermaid syntax
    mermaid_png = graph.get_graph().draw_mermaid_png()
    
    # Save as PNG
    with open("build_replanning_subgraph_visualization.png", "wb") as f:
        f.write(mermaid_png)
    
    print("Graph saved as 'build_replanning_subgraph_visualization.png'")
except Exception as e:
    print(f"Error generating PNG: {e}")
    # Fallback: print Mermaid syntax
    print("\nMermaid diagram syntax:")
    print(graph.get_graph().draw_mermaid())