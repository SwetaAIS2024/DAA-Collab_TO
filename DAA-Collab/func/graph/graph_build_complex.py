# -------------------------------
# Graph construction
# -------------------------------

# LangGraph minimal imports
from langgraph.graph import StateGraph, END
from func.state.agent_state import AgentState

from func.nodes.intent_classification_node import intent_classification
from func.nodes.metadata_extraction_node import metadata_extraction
from func.graph.subgraphs.planner_sg import build_planner_subgraph
from func.graph.subgraphs.execute_sg import build_execution_subgraph
from func.nodes.output_logging_node import output_formatter, logger

# Step 1: Train the ML model (one-time setup)
from utils.ML.intent_extraction_classification.ml_intent_classification import train_intent_classifier
train_intent_classifier()


# Step 2: Build the main graph
def build_graph():

    g = StateGraph(AgentState)

    # --- Add main nodes ---
    g.add_node("intent_classifier", intent_classification)
    g.add_node("metadata_extractor", metadata_extraction)

    # Subgraph nodes
    g.add_node("planner_subgraph", build_planner_subgraph())
    g.add_node("execution_subgraph", build_execution_subgraph())

    # Output nodes
    g.add_node("output_formatter", output_formatter)
    g.add_node("logger", logger)

    # ----- MAIN GRAPH EDGES ------
    g.set_entry_point("intent_classifier")
    g.add_edge("intent_classifier", "metadata_extractor")
    g.add_edge("metadata_extractor", "planner_subgraph")

    # Planner resolves tools + plan → goes to execution
    g.add_edge("planner_subgraph", "execution_subgraph")

    # Execution → output → logger → END
    g.add_edge("execution_subgraph", "output_formatter")
    g.add_edge("output_formatter", "logger")
    g.add_edge("logger", END)

    return g.compile()




# Build and compile the graph
graph = build_graph()

# Generate Mermaid diagram
try:
    # Get Mermaid syntax
    mermaid_png = graph.get_graph().draw_mermaid_png()
    
    # Save as PNG
    with open("graph_visualization.png", "wb") as f:
        f.write(mermaid_png)
    
    print("Graph saved as 'graph_visualization.png'")
except Exception as e:
    print(f"Error generating PNG: {e}")
    # Fallback: print Mermaid syntax
    print("\nMermaid diagram syntax:")
    print(graph.get_graph().draw_mermaid())