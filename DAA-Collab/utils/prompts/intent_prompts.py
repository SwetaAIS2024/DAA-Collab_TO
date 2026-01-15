"""
Intent Classification Prompts

This module contains all prompts related to intent extraction and classification.
Part of the centralized prompt management system.

Prompts:
- INTENT_EXTRACTION_SYSTEM_PROMPT: System prompt for LLM-based intent extraction
- INTENT_EXTRACTION_USER_PROMPT: User prompt template for intent extraction

Functions:
- load_intent_extraction_prompts(): Formats and loads intent extraction prompts with runtime data
"""

from typing import Dict, List, Tuple


INTENT_EXTRACTION_SYSTEM_PROMPT = """You are an intent extraction expert for traffic analysis systems.

Known intent types in the system:
{known_intent_classes}

Your task:
Known intent types ALREADY in the system (DO NOT extract these):
- visualization: Creating graphs, charts, plots, visual representations
- incident_detection: Finding, detecting, identifying accidents/incidents
- spatio_temporal: Location-based analysis, time-based analysis, date filtering
- meta_attributes: Data attributes, severity, type classification
- traffic_impact: Traffic flow analysis, congestion, impact assessment
- incident_classification: Categorizing incident types
- traffic_anomaly: Detecting unusual patterns

Full list: {known_intent_classes}

CRITICAL RULES:
1. The user query "get me the graph" → THIS IS 'visualization' (KNOWN)
2. "accident prone roads" → THIS IS 'incident_detection' (KNOWN)  
3. "timings" → THIS IS 'spatio_temporal' (KNOWN)
4. ONLY extract if request involves COMPLETELY NEW capabilities like:
   - "send email" → email_notification (NEW)
   - "generate PDF" → pdf_export (NEW)
   - "play sound alert" → audio_alert (NEW)

If ALL parts of the query can be satisfied by known intents, return [].

Return ONLY a JSON array of NEW capabilities not in the known list.

Now analyze:"""


INTENT_EXTRACTION_USER_PROMPT = """User prompt: "{prompt}"

Known intents already detected: {known_intents}

Unknown intents (JSON array):"""


def load_intent_extraction_prompts(
    prompt: str,
    known_intent_classes: List[str],
    known_intents: List[str]
) -> Tuple[str, str]:
    """
    Build system and user prompts for intent extraction
    
    Args:
        prompt: The user's query to analyze
        known_intent_classes: List of all known intent types in the system
        known_intents: List of intents already detected for this query
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = INTENT_EXTRACTION_SYSTEM_PROMPT.format(
        known_intent_classes=', '.join(known_intent_classes)
    )
    
    user_prompt = INTENT_EXTRACTION_USER_PROMPT.format(
        prompt=prompt,
        known_intents=known_intents if known_intents else "None"
    )
    
    return system_prompt, user_prompt
