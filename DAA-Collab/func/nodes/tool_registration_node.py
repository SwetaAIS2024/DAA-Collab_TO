
# # Auto-discover only MCP tool functions (decorated with @mcp.tool())
# legacy_tool_dictionary = {}
# for name, obj in inspect.getmembers(mcp_tools, inspect.isfunction):
#     # Only include functions actually defined in mcp_tools (exclude imports)
#     if not name.startswith('_') and hasattr(obj, '__module__') and obj.__module__ == 'mcp_tools':
#         # Optional: Further filter to only @mcp.tool() decorated functions
#         # Check if function has MCP tool markers (decorated functions often have special attributes)
#         # For now, we include all functions - helper functions won't break anything
#         legacy_tool_dictionary[name] = obj

# # TOOL_MAP = ToolMap(legacy_tool_dictionary)