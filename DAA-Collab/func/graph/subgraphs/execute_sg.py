# # -------------------------------
# # Execute node
# # -------------------------------

# # from func.state.agent_state import AgentState
# # from typing import List
# # # from utils.mcp_tools_registry. import TOOL_MAP

# # """Tool functions are now sourced from external mcp_tools.py."""
# # from utils.mcp_tools_registry.static_tools.mcp_tools import (
# #     load_and_analyze_csv,
# #     perform_advanced_eda_on_csv,
# #     generate_basic_plot,
# # )

# # TOOL_MAP = {
# #     "load_and_analyze_csv": load_and_analyze_csv,
# #     "perform_advanced_eda_on_csv": perform_advanced_eda_on_csv,
# #     "generate_basic_plot": generate_basic_plot,
# # }

# # def execute_node(state: AgentState) -> AgentState:
# #     outputs: List[str] = []
# #     for step in state.plan_steps:
# #         tool_name = step.get("tool")
# #         fn = TOOL_MAP.get(tool_name)
# #         if not fn:
# #             outputs.append(f"Skipped unknown tool: {tool_name}")
# #             continue
# #         try:
# #             out = fn(**step.get("args", {}))
# #         except Exception as e:
# #             err = f"Error executing {tool_name}: {e}"
# #             state.error_log.append(err)
# #             out = err
# #         outputs.append(out)
# #     state.raw_execution_output = "\n\n--- STEP END ---\n\n".join(outputs)
# #     return state


# """Execute node with MCP Protocol integration"""
# import asyncio
# from typing import List
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
# from func.state.agent_state import AgentState


# class MCPToolExecutor:
#     """MCP Protocol tool executor"""
    
#     def __init__(self, server_script_path: str = "utils/mcp_tools_registry/mcp_server.py"):
#         self.server_script_path = server_script_path
    
#     async def call_tool(self, tool_name: str, arguments: dict) -> str:
#         """Call tool via MCP protocol"""
#         try:
#             server_params = StdioServerParameters(
#                 command="python",
#                 args=[self.server_script_path],
#                 env=None
#             )
            
#             async with stdio_client(server_params) as (read, write):
#                 async with ClientSession(read, write) as session:
#                     await session.initialize()
#                     result = await session.call_tool(tool_name, arguments)
#                     return str(result.content[0].text) if result.content else "No output"
        
#         except Exception as e:
#             return f"MCP error: {str(e)}"


# def execute_node(state: AgentState) -> AgentState:
#     """Execute plan steps using MCP protocol"""
#     executor = MCPToolExecutor()
#     outputs: List[str] = []
    
#     for step in state.plan_steps:
#         tool_name = step.get("tool")
#         args = step.get("args", {})
        
#         try:
#             # Execute via MCP protocol
#             output = asyncio.run(executor.call_tool(tool_name, args))
#         except Exception as e:
#             output = f"Error executing {tool_name}: {e}"
#             state.error_log.append(output)
        
#         outputs.append(output)
    
#     state.raw_execution_output = "\n\n--- STEP END ---\n\n".join(outputs)
#     return state

from langgraph.graph import StateGraph, END
from func.state.agent_state import AgentState
from func.nodes.execute_sg_nodes.execution_node import execute_node, test_validation_node, user_feedback_node


def build_execution_subgraph():
    execution_subgraph =  StateGraph(AgentState) 

    execution_subgraph.add_node("execution", execute_node)
    execution_subgraph.add_node("test_validation", test_validation_node)
    execution_subgraph.add_node("user_feedback", user_feedback_node)

    execution_subgraph.set_entry_point("execution")

    execution_subgraph.add_edge("execution", "test_validation")

    # Conditional: feedback needed?
    execution_subgraph.add_conditional_edges(
        "test_validation",
        user_feedback_required,
        {
            False: "user_feedback",
            True: END
        }
    )

    execution_subgraph.add_edge("user_feedback", END)
    
    return execution_subgraph.compile()


def user_feedback_required()-> bool:
   
    condition_placeholder = True # Replace with actual condition logic
    return condition_placeholder



# Build and compile the graph
graph = build_execution_subgraph()

# Generate Mermaid diagram
try:
    # Get Mermaid syntax
    mermaid_png = graph.get_graph().draw_mermaid_png()
    
    # Save as PNG
    with open("build_execution_subgraph_visualization.png", "wb") as f:
        f.write(mermaid_png)
    
    print("Graph saved as 'build_execution_subgraph_visualization.png'")
except Exception as e:
    print(f"Error generating PNG: {e}")
    # Fallback: print Mermaid syntax
    print("\nMermaid diagram syntax:")
    print(graph.get_graph().draw_mermaid())