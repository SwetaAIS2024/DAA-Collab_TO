# Prompts Directory

This directory contains all LLM prompts used throughout the DAA-Collab application. Centralizing prompts here provides:

- **Version control**: Track prompt changes over time
- **Easy iteration**: Modify prompts without touching core logic
- **Consistency**: Single source of truth for all prompts
- **Testing**: Easier to A/B test different prompt variations

## Structure

```
prompts/
├── __init__.py                 # Main exports
├── README.md                   # This file
├── codegen_prompts.py          # Code generation & tool creation
├── intent_prompts.py           # (Future) Intent classification
├── planner_prompts.py          # (Future) Planning & orchestration
└── validation_prompts.py       # (Future) Code validation & review
```

## Current Prompts

### Code Generation (`codegen_prompts.py`)
- **CODEGEN_TOOL_GENERATION_PROMPT**: Main prompt for generating Python tool packages from specifications
  - Used by: `func/nodes/codegen_node.py`
  - Model: Qwen2.5-Coder or similar code-generation models
  - Purpose: Generate complete tool.py files with proper schemas and error handling

### Intent Extraction (`intent_prompts.py`)
- **INTENT_EXTRACTION_SYSTEM_PROMPT**: System prompt for extracting unknown intents from user queries
  - Used by: `utils/ML/ml_based_intent_classification/intent_classification.py`
  - Model: Qwen2.5 or similar instruction-following models
  - Purpose: Identify new capabilities not covered by existing known intent classes
- **INTENT_EXTRACTION_USER_PROMPT**: User prompt template for intent analysis
  - Purpose: Provide query context and known intents for analysis

## Adding New Prompts

1. Create a new file: `<category>_prompts.py`
2. Define prompt constants (use SCREAMING_SNAKE_CASE)
3. Add loader functions if prompt needs runtime formatting
4. Export from `__init__.py`
5. Update this README

Example:
```python
# new_category_prompts.py

MY_PROMPT_TEMPLATE = """
Your prompt here...
{variable_placeholder}
"""

def load_my_prompt(context: dict) -> str:
    return MY_PROMPT_TEMPLATE.format(**context)
```

## Best Practices

- Keep prompts as **template strings** with placeholders
- Provide **loader functions** for complex formatting logic
- Include **clear docstrings** explaining prompt purpose
- Version prompts if you need to maintain multiple variants
- Add **inline comments** for complex prompt sections
- Test prompt changes thoroughly before deployment
