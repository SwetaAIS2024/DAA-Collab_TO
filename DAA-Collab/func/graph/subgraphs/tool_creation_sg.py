

from langgraph.graph import StateGraph, END
from func.state.agent_state import AgentState
from func.nodes.tool_creation_sg_nodes.tool_creation_node import (
    input_preprocessing_node,
    tool_generation_node,
    tool_validation_node
)

def build_tool_creation_subgraph():
    tool_creation_subgraph =  StateGraph(AgentState)

    tool_creation_subgraph.add_node("input_preprocessing", input_preprocessing_node)
    tool_creation_subgraph.add_node("tool_generation", tool_generation_node)
    tool_creation_subgraph.add_node("tool_validation", tool_validation_node)

    tool_creation_subgraph.set_entry_point("input_preprocessing")

    tool_creation_subgraph.add_edge("input_preprocessing", "tool_generation")
    tool_creation_subgraph.add_edge("tool_generation", "tool_validation")

    # If validation fails → regenerate
    tool_creation_subgraph.add_conditional_edges(
        "tool_validation",
        tool_validation_success,
        {
            True: END,
            False: "tool_generation"
        }
    )

    return tool_creation_subgraph.compile()


# adding the helper condition function here 
def tool_validation_success(state: AgentState) -> bool:
    # Placeholder logic for tool validation success
    validation_passed = True  # Replace with actual validation logic
    return validation_passed


# Build and compile the graph
graph = build_tool_creation_subgraph()

# Generate Mermaid diagram
try:
    # Get Mermaid syntax
    mermaid_png = graph.get_graph().draw_mermaid_png()
    
    # Save as PNG
    with open("build_tool_creation_subgraph_visualization.png", "wb") as f:
        f.write(mermaid_png)
    
    print("Graph saved as 'build_tool_creation_subgraph_visualization.png'")
except Exception as e:
    print(f"Error generating PNG: {e}")
    # Fallback: print Mermaid syntax
    print("\nMermaid diagram syntax:")
    print(graph.get_graph().draw_mermaid())