from typing import List

def user_intent_extraction(user_instruction_prompt: str) -> List[str]:
    user_instruction = user_instruction_prompt.lower()
    intent_extraction_result = extraction_logic(user_instruction)
    return intent_extraction_result

def extraction_logic(instr: str) -> List[str]:
    # Placeholder for actual extraction logic
    intent_keywords = []
    return intent_keywords