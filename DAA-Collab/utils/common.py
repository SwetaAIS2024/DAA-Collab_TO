# -------------------------------
# Utility helpers
# -------------------------------

from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional


FOOTER_PATTERN = r"<!--output_json:(.+?)-->"

def make_seed_text(instruction: str, dataset_path: str) -> str:
    return f"Dataset: {dataset_path or 'unspecified'}\nInstructions: {instruction}"

def parse_footer(md: str) -> Dict[str, Any]:
    m = re.search(FOOTER_PATTERN, md, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}

def infer_plot_type(instr: str) -> str:
    low = instr.lower()
    if "histogram" in low: return "histogram"
    if "scatter" in low: return "scatterplot"
    if "box" in low: return "boxplot"
    if "pairplot" in low: return "pairplot"
    if "bar" in low or "chart" in low: return "bar_chart"
    # fallback
    return "histogram"

def infer_plot_columns(plot_type: str, numeric: List[str], categorical: List[str]) -> Optional[Dict[str, Any]]:
    if plot_type == "histogram" and numeric:
        return {"x_column": numeric[0]}
    if plot_type == "scatterplot" and len(numeric) >= 2:
        return {"x_column": numeric[0], "y_column": numeric[1]}
    if plot_type == "boxplot" and numeric and categorical:
        return {"x_column": numeric[0], "hue_column": categorical[0]}
    if plot_type == "bar_chart" and categorical:
        return {"x_column": categorical[0]}
    if plot_type == "pairplot" and len(numeric) >= 2:
        return {"columns_for_pairplot": numeric[:5]}
    return None


import re
from typing import Tuple, List

class PromptPreprocessor:
    """
    Comprehensive prompt preprocessing pipeline for intent classification.
    Handles negation, normalization, cleaning, and entity preservation.
    """
    
    def __init__(self):
        self.negation_patterns = [
            r'\b(do not|don\'t|does not|doesn\'t|did not|didn\'t)\s+(\w+\s+)?(\w+)',
            r'\b(not|no|never|without)\s+(\w+\s+)?(\w+)',
            r'\b(no)\s+(\w+)',
        ]
        
        # Common traffic domain abbreviations to preserve
        self.abbreviations = {
            'je': 'jurong east',
            'cte': 'central expressway',
            'pie': 'pan island expressway',
            'aye': 'ayer rajah expressway',
            'bke': 'bukit timah expressway',
            'ecp': 'east coast parkway',
            'kje': 'kranji expressway',
            'sle': 'seletar expressway',
            'tpe': 'tampines expressway',
            'cbd': 'central business district',
        }
    
    def remove_negated_terms(self, text: str) -> str:
        """
        Remove terms that appear in negation context.
        e.g., "do not detect anomalies" -> "do not"
        """
        cleaned_text = text.lower()
        
        for pattern in self.negation_patterns:
            matches = re.finditer(pattern, cleaned_text)
            for match in matches:
                full_match = match.group(0)
                negation_word = match.group(1)
                cleaned_text = cleaned_text.replace(full_match, negation_word)
        
        return cleaned_text
    
    def normalize_whitespace(self, text: str) -> str:
        """Remove extra whitespace and normalize spacing"""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def expand_abbreviations(self, text: str) -> str:
        """
        Expand common traffic domain abbreviations.
        e.g., "JE side" -> "jurong east side"
        """
        text_lower = text.lower()
        for abbr, full_form in self.abbreviations.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbr) + r'\b'
            text_lower = re.sub(pattern, full_form, text_lower)
        return text_lower
    
    def remove_filler_words(self, text: str) -> str:
        """
        Remove conversational filler words that don't add semantic value.
        e.g., "umm", "like", "you know"
        """
        filler_words = [
            r'\b(umm|uh|uhh|hmm|err|like|you know|i mean|basically|actually)\b',
        ]
        
        for pattern in filler_words:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    def normalize_punctuation(self, text: str) -> str:
        """
        Normalize punctuation for consistency.
        Replace multiple question marks/exclamations with single ones.
        """
        # Replace multiple punctuation
        text = re.sub(r'[?]+', '?', text)
        text = re.sub(r'[!]+', '!', text)
        text = re.sub(r'[.]+', '.', text)
        
        # Remove unnecessary punctuation at the end
        text = re.sub(r'[?.!]+\s*$', '', text)
        
        return text
    
    def extract_temporal_references(self, text: str) -> Tuple[str, List[str]]:
        """
        Extract and normalize temporal references.
        Returns: (cleaned_text, temporal_entities)
        """
        temporal_patterns = {
            'today': ['today', 'this day'],
            'yesterday': ['yesterday', 'last day'],
            'last_week': ['last week', 'past week', 'previous week'],
            'last_hour': ['last hour', 'past hour', 'previous hour'],
            'morning': ['morning', 'early hours', 'am'],
            'evening': ['evening', 'evening peak', 'pm'],
            'night': ['night', 'nighttime'],
        }
        
        temporal_entities = []
        for canonical, patterns in temporal_patterns.items():
            for pattern in patterns:
                if pattern in text.lower():
                    temporal_entities.append(canonical)
                    break
        
        return text, temporal_entities
    
    def extract_spatial_references(self, text: str) -> Tuple[str, List[str]]:
        """
        Extract and normalize spatial/location references.
        Returns: (cleaned_text, spatial_entities)
        """
        spatial_keywords = [
            'near', 'around', 'at', 'along', 'in', 'on',
            'side', 'sector', 'area', 'region', 'zone',
            'north', 'south', 'east', 'west',
            'northbound', 'southbound', 'eastbound', 'westbound'
        ]
        
        spatial_entities = []
        text_lower = text.lower()
        
        for keyword in spatial_keywords:
            if re.search(r'\b' + keyword + r'\b', text_lower):
                spatial_entities.append(keyword)
        
        return text, spatial_entities
    
    def preprocess(self, prompt: str, debug: bool = False) -> dict:
        """
        Main preprocessing pipeline.
        
        Args:
            prompt: Raw user input
            debug: Whether to print debug information
        
        Returns:
            dict with:
                - original: Original prompt
                - cleaned: Fully preprocessed prompt
                - temporal_entities: Extracted time references
                - spatial_entities: Extracted location references
                - processing_steps: List of applied transformations
        """
        steps = []
        
        # Step 1: Normalize whitespace
        text = self.normalize_whitespace(prompt)
        steps.append(("normalize_whitespace", text))
        
        # Step 2: Expand abbreviations
        text = self.expand_abbreviations(text)
        steps.append(("expand_abbreviations", text))
        
        # Step 3: Remove filler words
        text = self.remove_filler_words(text)
        steps.append(("remove_filler_words", text))
        
        # Step 4: Normalize punctuation
        text = self.normalize_punctuation(text)
        steps.append(("normalize_punctuation", text))
        
        # Step 5: Extract temporal references
        text, temporal_entities = self.extract_temporal_references(text)
        steps.append(("extract_temporal", text))
        
        # Step 6: Extract spatial references
        text, spatial_entities = self.extract_spatial_references(text)
        steps.append(("extract_spatial", text))
        
        # Step 7: Remove negated terms (LAST step to preserve negation context)
        text = self.remove_negated_terms(text)
        steps.append(("remove_negated_terms", text))
        
        # Step 8: Final whitespace normalization
        text = self.normalize_whitespace(text)
        
        result = {
            'original': prompt,
            'cleaned': text,
            'temporal_entities': list(set(temporal_entities)),
            'spatial_entities': list(set(spatial_entities)),
            'processing_steps': steps
        }
        
        if debug:
            print("\n" + "="*80)
            print("PROMPT PREPROCESSING PIPELINE")
            print("="*80)
            print(f"\nOriginal prompt:\n  {result['original']}")
            print(f"\nCleaned prompt:\n  {result['cleaned']}")
            print(f"\nTemporal entities: {result['temporal_entities']}")
            print(f"\nSpatial entities: {result['spatial_entities']}")
            print("\nProcessing steps:")
            for i, (step_name, step_result) in enumerate(result['processing_steps'], 1):
                print(f"  {i}. {step_name}")
                print(f"     -> {step_result}")
            print("="*80 + "\n")
        
        return result


# Singleton instance
_preprocessor = None

def get_preprocessor() -> PromptPreprocessor:
    """Get or create singleton preprocessor instance"""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = PromptPreprocessor()
    return _preprocessor


def preprocess_prompt(prompt: str, debug: bool = False) -> dict:
    """Convenience function for preprocessing"""
    preprocessor = get_preprocessor()
    return preprocessor.preprocess(prompt, debug=debug)








