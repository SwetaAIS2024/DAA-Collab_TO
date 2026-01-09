"""
Code Generator Node - Generates tools for missing capabilities
Handles: micro-planning + tool generation + sandbox validation + registry registration
"""

from typing import Dict, List, Optional, Any
from func.state.agent_state import AgentState
import json
import os
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

# Global variables for model caching
_codegen_model = None
_codegen_tokenizer = None
_codegen_device = None


def build_dataset_profile() -> Dict:
    """Build dataset profile for traffic incidents"""
    return {
        "id": "ds_incidents",
        "time_column": "timestamp",
        "columns": [
            {"name": "incident_id", "dtype": "string"},
            {"name": "timestamp", "dtype": "datetime"},
            {"name": "incident_type", "dtype": "string"},
            {"name": "severity", "dtype": "string"},
            {"name": "location", "dtype": "string"},
            {"name": "lanes_blocked", "dtype": "int"},
            {"name": "duration_min", "dtype": "float"}
        ],
        "notes": [
            "No explicit cost column present. Cost must be estimated.",
            "If lanes_blocked/duration_min missing, treat as 0."
        ]
    }


def build_registry_snapshot() -> Dict:
    """Build current tool registry snapshot"""
    # For MVP, return empty or minimal registry
    return {
        "snapshot_id": f"regsnap_local_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "existing_tools": [
            {
                "name": "detect_incidents",
                "purpose": "extract or filter incident records",
                "entrypoint": "tools.detect_incidents:run",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"}
            },
            {
                "name": "render_report",
                "purpose": "render report from summary tables",
                "entrypoint": "tools.render_report:run",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"}
            }
        ]
    }


def get_tool_contract_for_capability(capability: str, derived_requirements: Dict) -> Dict:
    """Get tool contract specification for a given capability"""
    
    # Define contracts for known capabilities
    contracts = {
        "incident_cost_estimation": {
            "name": "incident_cost_estimation",
            "purpose": "Estimate incident-level and aggregated costs for incidents in a time window.",
            "entrypoint_convention": "tool.py:run(input: dict) -> dict",
            "input_schema": {
                "type": "object",
                "required": ["dataset_ref", "time_window", "cost_model"],
                "properties": {
                    "dataset_ref": {"type": "string", "enum": ["ds_incidents"]},
                    "time_window": {"type": "string", "description": "e.g., last_week or ISO range"},
                    "cost_model": {
                        "type": "object",
                        "required": ["base_cost_by_severity", "per_minute_multiplier", "lane_block_multiplier"],
                        "properties": {
                            "base_cost_by_severity": {"type": "object"},
                            "per_minute_multiplier": {"type": "number"},
                            "lane_block_multiplier": {"type": "number"}
                        }
                    },
                    "group_by": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": ["incident_type", "severity"]
                    }
                }
            },
            "output_schema": {
                "type": "object",
                "required": ["summary", "by_group", "per_incident", "artifacts"],
                "properties": {
                    "summary": {
                        "type": "object",
                        "required": ["total_incidents", "total_estimated_cost", "avg_cost_per_incident"]
                    },
                    "by_group": {"type": "array", "items": {"type": "object"}},
                    "per_incident": {"type": "array", "items": {"type": "object"}},
                    "artifacts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["type", "path"],
                            "properties": {
                                "type": {"type": "string", "enum": ["csv", "json"]},
                                "path": {"type": "string"}
                            }
                        }
                    }
                }
            },
            "semantic_constraints": [
                "offline_only",
                "no_network_calls",
                "no_forecasting",
                "deterministic_given_same_inputs"
            ]
        },
        # Add more capability contracts as needed
        "aggregation_summary": {
            "name": "aggregation_summary",
            "purpose": "Aggregate and summarize incident data by specified dimensions",
            "entrypoint_convention": "tool.py:run(input: dict) -> dict",
            "input_schema": {
                "type": "object",
                "required": ["dataset_ref", "group_by", "aggregations"],
                "properties": {
                    "dataset_ref": {"type": "string"},
                    "group_by": {"type": "array", "items": {"type": "string"}},
                    "aggregations": {"type": "array", "items": {"type": "object"}}
                }
            },
            "output_schema": {
                "type": "object",
                "required": ["summary_table", "artifacts"]
            },
            "semantic_constraints": ["offline_only", "deterministic"]
        }
    }
    
    # Return contract if exists, otherwise create generic one
    if capability in contracts:
        return contracts[capability]
    
    # Generic contract for unknown capabilities
    return {
        "name": capability,
        "purpose": f"Tool for {capability} capability",
        "entrypoint_convention": "tool.py:run(input: dict) -> dict",
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
        "semantic_constraints": ["offline_only"]
    }


def build_codegen_llm_input(
    request_id: str,
    capability: str,
    user_query: str,
    intent_output: Dict,
    derived_requirements: Dict
) -> Dict:
    """
    Build the canonical CodeGen LLM input JSON
    This is what gets sent to the LLM to generate tool code
    """
    
    tool_contract = get_tool_contract_for_capability(capability, derived_requirements)
    
    codegen_input = {
        "request_meta": {
            "request_id": request_id,
            "source": "codegen_microplanner",
            "trigger": {
                "reason": "missing_capability",
                "capability": capability
            }
        },
        
        "context": {
            "user_query": user_query,
            "intent_output": intent_output,
            "derived_requirements": derived_requirements
        },
        
        "registry_snapshot": build_registry_snapshot(),
        
        "dataset_context": {
            "dataset_handles": [
                {
                    "id": "ds_incidents",
                    "path": "/mnt/data/incidents.csv",
                    "format": "csv"
                }
            ],
            "dataset_profile": build_dataset_profile()
        },
        
        "tool_contract": tool_contract,
        
        "sandbox_contract": {
            "language": "python",
            "python_version": "3.11",
            "allowed_packages": ["pandas", "numpy"],
            "forbidden_imports": ["requests", "subprocess"],
            "workspace_root": f"/mnt/data/tool_build/{request_id}"
        },
        
        "validation_contract": {
            "acceptance_tests": [
                {
                    "name": "smoke_schema_and_nonnegativity",
                    "input": build_test_input_for_capability(capability, derived_requirements),
                    "assertions": [
                        "output matches output_schema",
                        "all numeric outputs >= 0",
                        "no exceptions raised"
                    ]
                }
            ],
            "max_regenerate_attempts": 3
        },
        
        "expected_package": {
            "files": ["tool.py", "manifest.json", "tests.py"],
            "return_format": {
                "type": "object",
                "required": ["tool_manifest", "files", "how_to_run_tests"]
            }
        }
    }
    
    return codegen_input


def build_test_input_for_capability(capability: str, derived_requirements: Dict) -> Dict:
    """Build test input for a capability"""
    
    if capability == "incident_cost_estimation":
        return {
            "dataset_ref": "ds_incidents",
            "time_window": derived_requirements.get("time_window", "last_week"),
            "cost_model": {
                "base_cost_by_severity": {
                    "low": 100,
                    "medium": 500,
                    "high": 2000
                },
                "per_minute_multiplier": 5,
                "lane_block_multiplier": 1.2
            }
        }
    
    # Generic test input
    return {
        "dataset_ref": "ds_incidents",
        "time_window": derived_requirements.get("time_window", "last_week")
    }


def load_codegen_model():
    """Load Qwen Coder model for code generation with cascading fallback"""
    global _codegen_model, _codegen_tokenizer, _codegen_device
    
    if _codegen_model is not None:
        print("✅ CodeGen model already loaded")
        return _codegen_model, _codegen_tokenizer, _codegen_device
    
    # Determine device
    if torch.cuda.is_available():
        _codegen_device = "cuda"
        print(f"🖥️ Using device: {_codegen_device}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   VRAM Available: {total_vram:.1f} GB")
    else:
        _codegen_device = "cpu"
        print("🖥️ Using device: CPU (no CUDA available)")
    
    # Try models in order: 32B -> 7B -> simulation
    models_to_try = [
        ("Qwen/Qwen2.5-Coder-32B-Instruct", "32B"),
        ("Qwen/Qwen2.5-Coder-7B-Instruct", "7B")
    ]
    
    for model_name, model_size in models_to_try:
        try:
            print(f"🔄 Attempting to load: {model_name}")
            
            # Load tokenizer
            _codegen_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Load model with optimizations for memory
            _codegen_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if _codegen_device == "cuda" else torch.float32,
                device_map="auto" if _codegen_device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if _codegen_device == "cpu":
                _codegen_model = _codegen_model.to(_codegen_device)
            
            print(f"✅ CodeGen LLM model loaded successfully: {model_name}")
            
            # Show memory usage
            if _codegen_device == "cuda":
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f"   VRAM Allocated: {allocated:.2f} GB")
                print(f"   VRAM Reserved: {reserved:.2f} GB")
            
            return _codegen_model, _codegen_tokenizer, _codegen_device
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"❌ Out of memory error with {model_size} model")
            print(f"   Error: {e}")
            print(f"   Trying next smaller model...")
            
            # Clear memory
            if _codegen_model is not None:
                del _codegen_model
            if _codegen_tokenizer is not None:
                del _codegen_tokenizer
            _codegen_model = None
            _codegen_tokenizer = None
            
            if _codegen_device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            continue
            
        except Exception as e:
            print(f"❌ Error loading {model_size} model: {e}")
            print(f"   Trying next model...")
            
            # Clear memory
            if _codegen_model is not None:
                del _codegen_model
            if _codegen_tokenizer is not None:
                del _codegen_tokenizer
            _codegen_model = None
            _codegen_tokenizer = None
            
            if _codegen_device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            continue
    
    # All models failed
    print("❌ All models failed to load")
    print("   Falling back to simulation mode")
    return None, None, None


def build_codegen_prompt(codegen_input: Dict) -> str:
    """Build prompt for code generation LLM"""
    
    tool_contract = codegen_input["tool_contract"]
    dataset_profile = codegen_input["dataset_context"]["dataset_profile"]
    context = codegen_input["context"]
    
    prompt = f"""You are an expert Python developer. Generate a complete tool package for the following specification.

**User Query:** {context['user_query']}

**Tool Contract:**
- Name: {tool_contract['name']}
- Purpose: {tool_contract['purpose']}
- Input Schema: {json.dumps(tool_contract['input_schema'], indent=2)}
- Output Schema: {json.dumps(tool_contract['output_schema'], indent=2)}
- Constraints: {', '.join(tool_contract['semantic_constraints'])}

**Dataset Information:**
- Columns: {', '.join([col['name'] + ' (' + col['dtype'] + ')' for col in dataset_profile['columns']])}
- Time Column: {dataset_profile['time_column']}
- Notes: {'; '.join(dataset_profile['notes'])}

**Requirements:**
1. Generate a complete `tool.py` with a `run(input_data: dict) -> dict` function
2. Use only pandas and numpy (no network calls, no external APIs)
3. Handle missing data gracefully
4. Return data matching the output schema exactly
5. Include proper error handling
6. Add docstrings and comments

**Important:** Return ONLY the Python code for tool.py, no explanations.

```python
"""
    
    return prompt


def parse_generated_code(llm_response: str, capability: str) -> Dict:
    """Parse LLM response to extract code and create package"""
    
    # Extract code between ```python and ```
    code = llm_response
    if "```python" in llm_response:
        parts = llm_response.split("```python")
        if len(parts) > 1:
            code = parts[1].split("```")[0].strip()
    elif "```" in llm_response:
        parts = llm_response.split("```")
        if len(parts) > 1:
            code = parts[1].strip()
    
    # Create manifest
    manifest = {
        "name": capability,
        "version": "1.0.0",
        "description": f"Auto-generated tool for {capability}",
        "entrypoint": "tool.py:run",
        "runtime": "python3.11",
        "dependencies": ["pandas", "numpy"]
    }
    
    # Create basic tests
    tests = f"""# Auto-generated tests for {capability}
import json
from tool import run

def test_smoke():
    \"\"\"Basic smoke test\"\"\"
    test_input = {{
        "dataset_ref": "ds_incidents",
        "time_window": "last_week"
    }}
    try:
        result = run(test_input)
        assert isinstance(result, dict), "Output must be a dictionary"
        print("✅ Smoke test passed")
        return True
    except Exception as e:
        print(f"❌ Smoke test failed: {{e}}")
        return False

if __name__ == "__main__":
    success = test_smoke()
    exit(0 if success else 1)
"""
    
    return {
        "tool_manifest": manifest,
        "files": {
            "tool.py": code,
            "manifest.json": json.dumps(manifest, indent=2),
            "tests.py": tests
        },
        "how_to_run_tests": "python tests.py",
        "generation_metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "status": "generated"
        }
    }


def call_codegen_llm(codegen_input: Dict) -> Dict:
    """
    Call Qwen Coder LLM to generate tool code
    Falls back to simulation if model not available
    """
    
    capability = codegen_input["request_meta"]["trigger"]["capability"]
    
    # Try to load model
    model, tokenizer, device = load_codegen_model()
    
    if model is None or tokenizer is None:
        print("⚠️  Model not available, using simulation mode")
        return simulate_codegen_llm_call(codegen_input)
    
    try:
        # Build prompt
        prompt = build_codegen_prompt(codegen_input)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "cuda":
            inputs = inputs.to(device)
        
        # Generate
        print("🤖 Generating code...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract code (remove prompt)
        if prompt in generated_text:
            generated_code = generated_text.split(prompt)[1].strip()
        else:
            generated_code = generated_text
        
        print("✅ Code generation completed")
        
        # Parse and package
        return parse_generated_code(generated_code, capability)
        
    except Exception as e:
        print(f"❌ Error during code generation: {e}")
        print("   Falling back to simulation mode")
        return simulate_codegen_llm_call(codegen_input)
    
    finally:
        # Clear CUDA cache
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()


def simulate_codegen_llm_call(codegen_input: Dict) -> Dict:
    """
    Simulate LLM call to generate tool code
    In production, this would call an actual LLM
    For MVP, return a placeholder response
    """
    
    capability = codegen_input["request_meta"]["trigger"]["capability"]
    
    # Placeholder response
    return {
        "tool_manifest": {
            "name": capability,
            "version": "1.0.0",
            "purpose": codegen_input["tool_contract"]["purpose"],
            "entrypoint": "tool.py:run"
        },
        "files": {
            "tool.py": f"# Generated tool code for {capability}\ndef run(input_data):\n    return {{'status': 'placeholder'}}\n",
            "manifest.json": json.dumps({"name": capability, "version": "1.0.0"}),
            "tests.py": "# Placeholder tests\ndef test_smoke():\n    assert True\n"
        },
        "how_to_run_tests": "python tests.py",
        "generation_metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": "placeholder_mvp",
            "status": "simulated"
        }
    }


def validate_in_sandbox(generated_package: Dict) -> Dict:
    """
    Validate generated tool in sandbox
    For MVP, return success without actual execution
    """
    
    return {
        "validation_status": "simulated_success",
        "tests_passed": True,
        "schema_valid": True,
        "import_check": "passed",
        "security_check": "passed",
        "notes": "MVP: Actual sandbox validation not implemented yet"
    }


def register_tool(tool_manifest: Dict, validation_result: Dict) -> Dict:
    """
    Register tool in registry
    For MVP, just return confirmation
    """
    
    return {
        "registration_status": "simulated_success",
        "tool_id": f"tool_{tool_manifest['name']}_001",
        "registry_location": f"registry/{tool_manifest['name']}.json",
        "callable": True,
        "notes": "MVP: Actual registry not implemented yet"
    }


def codegen_node(state: AgentState) -> AgentState:
    """
    Code Generator Node - Generates tools for missing capabilities
    
    Flow:
    1. Takes MissingToolRequests from planner
    2. For each request, builds CodeGen LLM input JSON
    3. Calls LLM to generate tool code (simulated for MVP)
    4. Validates in sandbox (simulated for MVP)
    5. Registers tool (simulated for MVP)
    """
    
    print("\n" + "="*80)
    print("🛠️ CODE GENERATOR NODE")
    print("="*80)
    
    missing_tool_requests = getattr(state, 'missing_tool_requests', [])
    
    if not missing_tool_requests:
        print("No missing tool requests. Skipping CodeGen.")
        state.codegen_status = "skipped"  # type: ignore
        return state
    
    print(f"Processing {len(missing_tool_requests)} tool requests")
    
    generated_tools = []
    
    for idx, request in enumerate(missing_tool_requests, 1):
        capability = request["capability"]
        print(f"\n--- Tool Request {idx}/{len(missing_tool_requests)} ---")
        print(f"Capability: {capability}")
        
        # Step 1: Build CodeGen LLM input
        request_id = f"cg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx:04d}"
        
        codegen_input = build_codegen_llm_input(
            request_id=request_id,
            capability=capability,
            user_query=request["context"]["user_query"],
            intent_output=request["context"]["intent_output"],
            derived_requirements=request["derived_requirements"]
        )
        
        print(f"✅ Built CodeGen LLM input (request_id: {request_id})")
        
        # Save CodeGen input for debugging (optional)
        output_dir = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "data", "codegen_requests"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        input_file = os.path.join(output_dir, f"{request_id}_input.json")
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(codegen_input, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved CodeGen input to: {input_file}")
        
        # Step 2: Call LLM (now using real Qwen Coder model)
        print("🤖 Calling CodeGen LLM...")
        generated_package = call_codegen_llm(codegen_input)
        print(f"✅ Tool package generated")
        
        # Save generated code for inspection
        code_file = os.path.join(output_dir, f"{request_id}_tool.py")
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(generated_package["files"]["tool.py"])
        print(f"💾 Saved generated tool to: {code_file}")
        
        # Step 3: Validate in sandbox (simulated for MVP)
        print("🔒 Validating in sandbox (simulated)...")
        validation_result = validate_in_sandbox(generated_package)
        print(f"✅ Validation: {validation_result['validation_status']}")
        
        # Step 4: Register tool (simulated for MVP)
        print("📝 Registering tool (simulated)...")
        registration_result = register_tool(
            generated_package["tool_manifest"],
            validation_result
        )
        print(f"✅ Tool registered: {registration_result['tool_id']}")
        
        # Store results
        generated_tools.append({
            "request_id": request_id,
            "capability": capability,
            "codegen_input": codegen_input,
            "generated_package": generated_package,
            "validation": validation_result,
            "registration": registration_result
        })
    
    # Update state
    state.generated_tools = generated_tools  # type: ignore
    state.codegen_status = "completed"  # type: ignore
    
    print(f"\n✅ CodeGen completed. {len(generated_tools)} tools generated.")
    print("="*80)
    
    return state
