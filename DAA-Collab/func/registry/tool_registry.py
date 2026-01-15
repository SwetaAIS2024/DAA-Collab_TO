"""
Tool Registry - Persistent storage and retrieval of generated tools
Enables the "closing the loop" pattern: failures → tools → capabilities
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path


class ToolRegistry:
    """
    Persistent registry for auto-generated tools
    Stores: manifests, code, metadata, capability mappings
    """
    
    def __init__(self, registry_root: str = None):
        if registry_root is None:
            registry_root = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "data", "tool_registry"
            )
        
        self.registry_root = Path(registry_root)
        self.registry_root.mkdir(parents=True, exist_ok=True)
        
        # Registry structure:
        # tool_registry/
        #   ├── index.json              # Master index of all tools
        #   ├── capability_map.json      # Capability → Tool mappings
        #   └── tools/
        #       ├── incident_cost_estimation_v1/
        #       │   ├── manifest.json
        #       │   ├── tool.py
        #       │   ├── tests.py
        #       │   └── metadata.json
        #       └── aggregation_summary_v1/
        #           └── ...
        
        self.tools_dir = self.registry_root / "tools"
        self.tools_dir.mkdir(exist_ok=True)
        
        self.index_file = self.registry_root / "index.json"
        self.capability_map_file = self.registry_root / "capability_map.json"
        
        # Initialize if needed
        if not self.index_file.exists():
            self._init_index()
        if not self.capability_map_file.exists():
            self._init_capability_map()
    
    def _init_index(self):
        """Initialize empty registry index"""
        index = {
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "tools": [],
            "total_count": 0
        }
        self.index_file.write_text(json.dumps(index, indent=2))
    
    def _init_capability_map(self):
        """Initialize empty capability map"""
        capability_map = {
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "mappings": {}
        }
        self.capability_map_file.write_text(json.dumps(capability_map, indent=2))
    
    def _load_index(self) -> Dict:
        """Load registry index"""
        return json.loads(self.index_file.read_text())
    
    def _save_index(self, index: Dict):
        """Save registry index"""
        self.index_file.write_text(json.dumps(index, indent=2))
    
    def _load_capability_map(self) -> Dict:
        """Load capability map"""
        return json.loads(self.capability_map_file.read_text())
    
    def _save_capability_map(self, capability_map: Dict):
        """Save capability map"""
        self.capability_map_file.write_text(json.dumps(capability_map, indent=2))
    
    def register_tool(
        self,
        capability: str,
        generated_package: Dict,
        validation_result: Dict,
        request_metadata: Dict
    ) -> Dict:
        """
        Register a generated tool (promotes from temp → permanent)
        
        Returns:
            {
                "tool_id": str,
                "registry_path": str,
                "callable": bool,
                "capability": str
            }
        """
        
        # Generate tool ID
        index = self._load_index()
        version = self._get_next_version(capability, index)
        tool_id = f"{capability}_v{version}"
        
        # Create tool directory
        tool_dir = self.tools_dir / tool_id
        tool_dir.mkdir(exist_ok=True)
        
        # Save files
        for filename, content in generated_package["files"].items():
            file_path = tool_dir / filename
            file_path.write_text(content)
        
        # Create metadata file
        metadata = {
            "tool_id": tool_id,
            "capability": capability,
            "version": version,
            "registered_at": datetime.now().isoformat(),
            "request_metadata": request_metadata,
            "validation_result": validation_result,
            "generation_metadata": generated_package.get("generation_metadata", {}),
            "manifest": generated_package["tool_manifest"],
            "status": "active",
            "usage_count": 0,
            "last_used": None
        }
        
        metadata_file = tool_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))
        
        # Update index
        index["tools"].append({
            "tool_id": tool_id,
            "capability": capability,
            "version": version,
            "registered_at": metadata["registered_at"],
            "path": str(tool_dir.relative_to(self.registry_root)),
            "status": "active"
        })
        index["total_count"] = len(index["tools"])
        self._save_index(index)
        
        # Update capability map
        capability_map = self._load_capability_map()
        if capability not in capability_map["mappings"]:
            capability_map["mappings"][capability] = []
        
        capability_map["mappings"][capability].append({
            "tool_id": tool_id,
            "version": version,
            "registered_at": metadata["registered_at"],
            "is_latest": True
        })
        
        # Mark previous versions as not latest
        for entry in capability_map["mappings"][capability][:-1]:
            entry["is_latest"] = False
        
        self._save_capability_map(capability_map)
        
        print(f"✅ Tool registered: {tool_id}")
        print(f"   Path: {tool_dir}")
        print(f"   Capability: {capability}")
        
        return {
            "tool_id": tool_id,
            "registry_path": str(tool_dir),
            "callable": True,
            "capability": capability,
            "version": version
        }
    
    def _get_next_version(self, capability: str, index: Dict) -> int:
        """Get next version number for a capability"""
        existing_versions = [
            tool["version"] 
            for tool in index["tools"] 
            if tool["capability"] == capability
        ]
        return max(existing_versions, default=0) + 1
    
    def get_tool_for_capability(self, capability: str) -> Optional[Dict]:
        """
        Find latest tool for a given capability
        Returns None if no tool exists
        """
        capability_map = self._load_capability_map()
        
        if capability not in capability_map["mappings"]:
            return None
        
        # Get latest version
        latest = None
        for entry in capability_map["mappings"][capability]:
            if entry.get("is_latest", False):
                latest = entry
                break
        
        if not latest:
            return None
        
        # Load full metadata
        tool_dir = self.tools_dir / latest["tool_id"]
        metadata_file = tool_dir / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        metadata = json.loads(metadata_file.read_text())
        
        # Load tool code
        tool_file = tool_dir / "tool.py"
        if tool_file.exists():
            metadata["tool_code"] = tool_file.read_text()
        
        return metadata
    
    def has_capability(self, capability: str) -> bool:
        """Check if a tool exists for this capability"""
        capability_map = self._load_capability_map()
        return capability in capability_map["mappings"]
    
    def list_all_capabilities(self) -> List[str]:
        """List all registered capabilities"""
        capability_map = self._load_capability_map()
        return list(capability_map["mappings"].keys())
    
    def get_registry_snapshot(self) -> Dict:
        """
        Get current registry state for planner/codegen context
        This replaces the hardcoded build_registry_snapshot()
        """
        index = self._load_index()
        capability_map = self._load_capability_map()
        
        existing_tools = []
        for tool_entry in index["tools"]:
            if tool_entry["status"] == "active":
                tool_dir = self.tools_dir / tool_entry["tool_id"]
                metadata_file = tool_dir / "metadata.json"
                
                if metadata_file.exists():
                    metadata = json.loads(metadata_file.read_text())
                    existing_tools.append({
                        "name": metadata["manifest"]["name"],
                        "purpose": metadata["manifest"].get("purpose", ""),
                        "entrypoint": metadata["manifest"]["entrypoint"],
                        "capability": metadata["capability"],
                        "version": metadata["version"],
                        "tool_id": tool_entry["tool_id"]
                    })
        
        return {
            "snapshot_id": f"regsnap_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "total_tools": index["total_count"],
            "total_capabilities": len(capability_map["mappings"]),
            "existing_tools": existing_tools
        }
    
    def increment_usage(self, tool_id: str):
        """Track tool usage (for analytics/deprecation)"""
        tool_dir = self.tools_dir / tool_id
        metadata_file = tool_dir / "metadata.json"
        
        if not metadata_file.exists():
            return
        
        metadata = json.loads(metadata_file.read_text())
        metadata["usage_count"] = metadata.get("usage_count", 0) + 1
        metadata["last_used"] = datetime.now().isoformat()
        
        metadata_file.write_text(json.dumps(metadata, indent=2))
