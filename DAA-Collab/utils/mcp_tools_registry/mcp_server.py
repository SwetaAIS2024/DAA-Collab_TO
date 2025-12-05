"""
MCP Server for Data Analysis Tools
Implements Model Context Protocol specification
"""
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Add parent directories to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

# Import the actual synchronous functions from mcp_tools.py
try:
    from static_tools.mcp_tools import (
        load_and_analyze_csv,
        perform_advanced_eda_on_csv,
        generate_basic_plot,
    )
except ImportError:
    # Fallback import path
    from utils.mcp_tools_registry.static_tools.mcp_tools import (
        load_and_analyze_csv,
        perform_advanced_eda_on_csv,
        generate_basic_plot,
    )

# Initialize MCP Server
app = Server("data-analysis-tools")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools following MCP protocol"""
    return [
        Tool(
            name="load_and_analyze_csv",
            description="Load CSV and provide detailed summary of structure and contents. ALWAYS use this as the FIRST tool.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the CSV file to analyze"
                    },
                    "auto_expand_json": {
                        "type": "boolean",
                        "description": "Automatically detect and expand JSON columns",
                        "default": True
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="perform_advanced_eda_on_csv",
            description="Perform comprehensive exploratory data analysis with correlation, clustering, and regression",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the CSV file"
                    },
                    "auto_detect_analysis": {
                        "type": "boolean",
                        "description": "Automatically detect and perform appropriate analysis",
                        "default": False
                    },
                    "auto_expand_json": {
                        "type": "boolean",
                        "description": "Expand JSON columns",
                        "default": True
                    },
                    "n_clusters": {
                        "type": "integer",
                        "description": "Number of clusters for KMeans"
                    },
                    "random_state": {
                        "type": "integer",
                        "description": "Random state for reproducibility",
                        "default": 0
                    },
                    "n_init": {
                        "type": "string",
                        "description": "KMeans n_init parameter",
                        "default": "auto"
                    },
                    "x_column": {
                        "type": "string",
                        "description": "Independent variable for regression"
                    },
                    "y_column": {
                        "type": "string",
                        "description": "Dependent variable for regression"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="generate_basic_plot",
            description="Generate visualization plots: histogram, scatterplot, boxplot, pairplot, or bar_chart",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the CSV file"
                    },
                    "plot_type": {
                        "type": "string",
                        "enum": ["histogram", "scatterplot", "boxplot", "pairplot", "bar_chart"],
                        "description": "Type of plot to generate"
                    },
                    "x_column": {
                        "type": "string",
                        "description": "X-axis column name"
                    },
                    "y_column": {
                        "type": "string",
                        "description": "Y-axis column name"
                    },
                    "hue_column": {
                        "type": "string",
                        "description": "Grouping column for boxplot/bar_chart"
                    },
                    "columns_for_pairplot": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns for pairplot"
                    },
                    "title": {
                        "type": "string",
                        "description": "Plot title",
                        "default": "Generated Plot"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for plot",
                        "default": "."
                    }
                },
                "required": ["file_path", "plot_type"]
            }
        )
    ]


async def run_sync_in_executor(func, *args, **kwargs):
    """Run synchronous function in executor to avoid blocking event loop"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute tool calls following MCP protocol"""
    
    try:
        if name == "load_and_analyze_csv":
            # Run synchronous function in executor
            result = await run_sync_in_executor(
                load_and_analyze_csv,
                arguments.get("file_path"),
                arguments.get("auto_expand_json", True)
            )
            return [TextContent(type="text", text=str(result))]
        
        elif name == "perform_advanced_eda_on_csv":
            # Run synchronous function in executor
            result = await run_sync_in_executor(
                perform_advanced_eda_on_csv,
                arguments.get("file_path"),
                arguments.get("auto_detect_analysis", False),
                arguments.get("auto_expand_json", True),
                arguments.get("n_clusters"),
                arguments.get("random_state", 0),
                arguments.get("n_init", "auto"),
                arguments.get("x_column"),
                arguments.get("y_column")
            )
            return [TextContent(type="text", text=str(result))]
        
        elif name == "generate_basic_plot":
            # Run synchronous function in executor
            result = await run_sync_in_executor(
                generate_basic_plot,
                arguments.get("file_path"),
                arguments.get("plot_type"),
                arguments.get("x_column"),
                arguments.get("y_column"),
                arguments.get("hue_column"),
                arguments.get("columns_for_pairplot"),
                arguments.get("title", "Generated Plot"),
                arguments.get("output_dir", ".")
            )
            return [TextContent(type="text", text=str(result))]
        
        else:
            error_msg = f"Unknown tool: {name}"
            return [TextContent(type="text", text=error_msg)]
    
    except Exception as e:
        import traceback
        error_msg = f"Error executing {name}:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return [TextContent(type="text", text=error_msg)]


async def main():
    """Run the MCP server via stdio"""
    try:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
    except Exception as e:
        import traceback
        print(f"MCP Server error: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise


if __name__ == "__main__":
    asyncio.run(main())