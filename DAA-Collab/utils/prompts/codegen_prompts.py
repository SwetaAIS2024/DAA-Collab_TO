"""
Code Generation Prompts

This module contains all prompts related to automatic code generation and tool creation.
Part of the centralized prompt management system.

Prompts:
- CODEGEN_TOOL_GENERATION_PROMPT: Main prompt for generating tool packages from specifications

Functions:
- load_codegen_prompt(): Formats and loads the code generation prompt with runtime data
"""

import json
from typing import Dict


CODEGEN_TOOL_GENERATION_PROMPT = """You are an expert Python developer. Generate a complete tool package for the following specification.

**User Query:** {user_query}

**Tool Specification:**
- Name: {tool_name}
- Purpose: {tool_purpose}
- Input Schema: {input_schema}
- Output Schema: {output_schema}
- Constraints: {constraints}

**Dataset Information:**
- Columns: {dataset_columns}
- Time Column: {time_column}
- Notes: {dataset_notes}

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


def load_codegen_prompt(codegen_input: Dict) -> str:
    """
    Build prompt for code generation LLM from input specification
    
    Args:
        codegen_input: Dictionary containing tool_spec, dataset_context, and context
        
    Returns:
        Formatted prompt string
    """
    tool_spec = codegen_input["tool_spec"]
    dataset_profile = codegen_input["dataset_context"]["dataset_profile"]
    context = codegen_input["context"]
    
    # Format dataset columns
    dataset_columns = ', '.join([
        f"{col['name']} ({col['dtype']})" 
        for col in dataset_profile['columns']
    ])
    
    # Format constraints
    constraints = ', '.join(tool_spec['semantic_constraints'])
    
    # Build the prompt
    prompt = CODEGEN_TOOL_GENERATION_PROMPT.format(
        user_query=context['user_query'],
        tool_name=tool_spec['name'],
        tool_purpose=tool_spec['purpose'],
        input_schema=json.dumps(tool_spec['input_schema'], indent=2),
        output_schema=json.dumps(tool_spec['output_schema'], indent=2),
        constraints=constraints,
        dataset_columns=dataset_columns,
        time_column=dataset_profile['time_column'],
        dataset_notes='; '.join(dataset_profile['notes'])
    )
    
    return prompt
