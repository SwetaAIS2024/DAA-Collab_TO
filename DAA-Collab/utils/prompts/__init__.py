"""
Prompts Module
Centralized storage for all LLM prompts used across the application

Prompt Categories:
- codegen_prompts: Code generation and tool creation
- intent_prompts: Intent classification and understanding
- planner_prompts: Planning and orchestration (future)
- validation_prompts: Code validation and review (future)
"""

from .codegen_prompts import (
    CODEGEN_TOOL_GENERATION_PROMPT,
    load_codegen_prompt
)

from .intent_prompts import (
    INTENT_EXTRACTION_SYSTEM_PROMPT,
    INTENT_EXTRACTION_USER_PROMPT,
    load_intent_extraction_prompts
)

__all__ = [
    # Code generation
    'CODEGEN_TOOL_GENERATION_PROMPT',
    'load_codegen_prompt',
    
    # Intent extraction
    'INTENT_EXTRACTION_SYSTEM_PROMPT',
    'INTENT_EXTRACTION_USER_PROMPT',
    'load_intent_extraction_prompts',
]
