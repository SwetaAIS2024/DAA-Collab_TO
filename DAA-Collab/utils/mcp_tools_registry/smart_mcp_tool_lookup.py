"""
Smart Tool Map - Lazy loading tool registry for MCP tools.

This module provides a dictionary subclass that automatically loads tools
from the generated_tools/ directory when they are requested, enabling
dynamic tool discovery without breaking legacy code.
"""

from pathlib import Path
import importlib.util
import sys
from typing import Optional, Callable, Any
import json


class ToolMap(dict):
    """
    A dictionary that automatically loads tools from generated_tools/ directory
    when they are accessed but not yet in the cache.
    
    Features:
    - Lazy loading: Tools only loaded when requested
    - Caching: Once loaded, tools are cached for fast subsequent access
    - Backward compatible: Works exactly like a regular dict for existing tools
    - Scalable: Can handle 1000s of tools without manual registration
    
    Usage:
        TOOL_MAP = ToolMap({
            "existing_tool": existing_tool_function,
            ...
        })
        
        # Later, when a generated tool is needed:
        fn = TOOL_MAP.get("new_generated_tool")  # Auto-loads if exists
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generated_tools_dir = Path(__file__).parent.parent / "generated_tools"
        self.registry_file = self.generated_tools_dir / "_registry.json"
        self._load_registry()
    
    def _load_registry(self):
        """Load the registry file that tracks all generated tools."""
        self._registry = {}
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    self._registry = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load tool registry: {e}")
    
    def __getitem__(self, key: str) -> Callable:
        """
        Get a tool by name. If not in cache, try loading from generated_tools/.
        
        Args:
            key: Tool name to retrieve
            
        Returns:
            Tool function
            
        Raises:
            KeyError: If tool not found anywhere
        """
        # Check if already cached
        if key in self.keys():
            return super().__getitem__(key)
        
        # Try loading from generated_tools/
        tool = self._load_from_generated(key)
        if tool:
            # Cache it for future access
            self[key] = tool
            return tool
        
        # Not found anywhere
        raise KeyError(f"Tool '{key}' not found in TOOL_MAP or generated_tools/")
    
    def get(self, key: str, default: Optional[Any] = None) -> Optional[Callable]:
        """
        Get a tool by name with a default fallback.
        
        Args:
            key: Tool name to retrieve
            default: Value to return if tool not found
            
        Returns:
            Tool function or default value
        """
        try:
            return self.__getitem__(key)
        except KeyError:
            return default
    
    def _load_from_generated(self, tool_name: str) -> Optional[Callable]:
        """
        Try to load a tool from the generated_tools/ directory.
        
        Args:
            tool_name: Name of the tool to load
            
        Returns:
            Tool function if found, None otherwise
        """
        # Check registry first for metadata
        if tool_name in self._registry:
            tool_info = self._registry[tool_name]
            # Check if tool is approved (if approval system is used)
            if not tool_info.get("approved", True):
                print(f"Warning: Tool '{tool_name}' exists but is not approved")
                return None
            filename = tool_info.get("filename", f"{tool_name}.py")
        else:
            filename = f"{tool_name}.py"
        
        tool_file = self.generated_tools_dir / filename
        
        # Check if file exists
        if not tool_file.exists():
            return None
        
        try:
            # Dynamic import using importlib
            spec = importlib.util.spec_from_file_location(tool_name, tool_file)
            if spec is None or spec.loader is None:
                print(f"Error: Could not create module spec for {tool_name}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            
            # Add to sys.modules to handle imports within the tool
            sys.modules[f"generated_tools.{tool_name}"] = module
            
            # Execute the module
            spec.loader.exec_module(module)
            
            # Get the function from the module (assumes function has same name as file)
            if hasattr(module, tool_name):
                print(f"✓ Loaded tool '{tool_name}' from generated_tools/")
                return getattr(module, tool_name)
            else:
                print(f"Warning: File {filename} exists but doesn't contain function '{tool_name}'")
                return None
                
        except Exception as e:
            print(f"Error loading tool '{tool_name}': {e}")
            return None
    
    def register_tool(self, tool_name: str, tool_func: Callable, metadata: Optional[dict] = None):
        """
        Manually register a tool (useful when generating tools at runtime).
        
        Args:
            tool_name: Name of the tool
            tool_func: The tool function
            metadata: Optional metadata (version, created_at, etc.)
        """
        # Add to cache immediately
        self[tool_name] = tool_func
        
        # Update registry if metadata provided
        if metadata:
            if not self._registry:
                self._registry = {}
            self._registry[tool_name] = metadata
            self._save_registry()
    
    def _save_registry(self):
        """Save the registry to disk."""
        try:
            self.generated_tools_dir.mkdir(exist_ok=True)
            with open(self.registry_file, 'w') as f:
                json.dump(self._registry, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save tool registry: {e}")
    
    def list_available_tools(self) -> dict:
        """
        List all available tools (cached + in generated_tools/).
        
        Returns:
            Dict with 'cached' and 'available' tool lists
        """
        cached = list(self.keys())
        
        # Scan generated_tools directory
        available = []
        if self.generated_tools_dir.exists():
            for file in self.generated_tools_dir.glob("*.py"):
                if file.stem != "__init__" and not file.stem.startswith("_"):
                    available.append(file.stem)
        
        return {
            "cached": cached,
            "available_in_generated": [t for t in available if t not in cached],
            "total": len(set(cached + available))
        }


def get_tool(tool_map: ToolMap, tool_name: str) -> Optional[Callable]:
    """
    Helper function to get a tool from ToolMap.
    
    This is a convenience function if you prefer functional style.
    
    Args:
        tool_map: The ToolMap instance
        tool_name: Name of the tool to retrieve
        
    Returns:
        Tool function or None if not found
    """
    return tool_map.get(tool_name)