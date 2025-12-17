# -------------------------------
# Utility helpers
# -------------------------------

from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional, Tuple


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


class PromptPreprocessor:
    """
    Comprehensive prompt preprocessing pipeline for intent classification.
    Handles negation, normalization, cleaning, and entity preservation.
    """
    
    def __init__(self):
        # Enhanced negation patterns - ordered by specificity
        self.negation_patterns = [
            r'\b(do\s+not|don\'t|does\s+not|doesn\'t|did\s+not|didn\'t)\s+(\w+(?:\s+\w+){0,2})',
            r'\b(should\s+not|shouldn\'t|would\s+not|wouldn\'t|could\s+not|couldn\'t)\s+(\w+(?:\s+\w+){0,2})',
            r'\b(not|never|without)\s+(\w+(?:\s+\w+){0,2})',
            r'\bno\s+(\w+(?:\s+\w+){0,1})',
            r'\b(exclude|omit|skip|ignore|avoid)\s+(\w+(?:\s+\w+){0,2})'
        ]
        
        # Enhanced abbreviations (traffic domain + general)
        self.abbreviations = {
            # Traffic locations (Singapore specific)
            r'\bje\b': 'jurong east',
            r'\bcte\b': 'central expressway',
            r'\bpie\b': 'pan island expressway',
            r'\baye\b': 'ayer rajah expressway',
            r'\bbke\b': 'bukit timah expressway',
            r'\becp\b': 'east coast parkway',
            r'\bkje\b': 'kranji expressway',
            r'\bsle\b': 'seletar expressway',
            r'\btpe\b': 'tampines expressway',
            r'\bcbd\b': 'central business district',
            
            # General road types
            r'\bhwy\b': 'highway',
            r'\brd\b': 'road',
            r'\bst\b': 'street',
            r'\bave\b': 'avenue',
            r'\bblvd\b': 'boulevard',
            r'\bpkwy\b': 'parkway',
            r'\bln\b': 'lane',
            r'\bdr\b': 'drive',
            
            # Directions
            r'\bn\b': 'north',
            r'\bs\b': 'south',
            r'\be\b': 'east',
            r'\bw\b': 'west',
            r'\bne\b': 'northeast',
            r'\bnw\b': 'northwest',
            r'\bse\b': 'southeast',
            r'\bsw\b': 'southwest',
            
            # File formats & technical
            r'\bpdf\b': 'portable document format',
            r'\bcsv\b': 'comma separated values',
            r'\bjson\b': 'javascript object notation',
            r'\bxml\b': 'extensible markup language',
            
            # Time
            r'\bam\b': 'morning',
            r'\bpm\b': 'afternoon',
            
            # Common
            r'\bvs\b': 'versus',
            r'\betc\b': 'et cetera',
        }
        
        # Filler words (conversational noise)
        self.filler_words = {
            'umm', 'uh', 'uhh', 'hmm', 'err', 'like', 
            'you know', 'i mean', 'basically', 'actually',
            'kind of', 'sort of', 'really', 'very', 'just',
            'literally', 'totally', 'absolutely'
        }
        
        # Enhanced temporal keywords
        self.temporal_keywords = {
            # Absolute
            'today', 'yesterday', 'tomorrow', 'tonight', 'now', 'current', 'currently',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'weekend', 'weekday',
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            
            # Relative
            'recent', 'recently', 'latest', 'last', 'next', 'previous', 'upcoming', 'past',
            'morning', 'afternoon', 'evening', 'night', 'dawn', 'dusk', 'noon', 'midnight',
            'week', 'month', 'year', 'day', 'hour', 'minute', 'second',
            'weekly', 'monthly', 'yearly', 'daily', 'hourly',
            
            # Ranges
            'between', 'during', 'from', 'to', 'until', 'before', 'after', 'since',
            'within', 'over', 'throughout', 'at'
        }
        
        # Enhanced spatial keywords
        self.spatial_keywords = {
            # Locations
            'highway', 'expressway', 'road', 'street', 'avenue', 'boulevard', 'lane', 'parkway',
            'intersection', 'junction', 'exit', 'ramp', 'bridge', 'tunnel', 'overpass', 'underpass',
            
            # Directions
            'north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest',
            'northbound', 'southbound', 'eastbound', 'westbound',
            'left', 'right', 'straight', 'ahead',
            
            # Areas
            'area', 'region', 'zone', 'district', 'sector', 'side', 'route', 'corridor',
            'downtown', 'uptown', 'midtown', 'suburb', 'city', 'town', 'village',
            
            # Prepositions (spatial context)
            'on', 'at', 'near', 'along', 'around', 'between', 'across', 'through', 
            'via', 'toward', 'towards', 'from', 'to', 'by', 'beside', 'next'
        }
    
    def remove_negated_terms(self, text: str) -> Tuple[str, List[str]]:
        """
        Remove terms in negation context and return removed terms.
        Examples:
            "do not detect anomalies" -> ("", ["detect anomalies"])
            "show incidents but exclude minor ones" -> ("show incidents but", ["minor ones"])
        """
        removed_terms = []
        cleaned = text
        
        for pattern in self.negation_patterns:
            matches = list(re.finditer(pattern, cleaned, re.IGNORECASE))
            for match in reversed(matches):  # Reverse to avoid index shifting
                # Extract the negated term (last captured group)
                negated_term = match.groups()[-1] if match.groups() else match.group(0)
                removed_terms.append(negated_term)
                # Replace full match with empty string
                cleaned = cleaned[:match.start()] + cleaned[match.end():]
        
        # Clean up extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned, removed_terms
    
    def normalize_whitespace(self, text: Optional[str]) -> str:
        """Remove extra whitespace and normalize spacing"""
        if text is None:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def expand_abbreviations(self, text: str) -> str:
        """
        Expand domain-specific and general abbreviations.
        Preserves case context where needed.
        """
        for abbr_pattern, full_form in self.abbreviations.items():
            text = re.sub(abbr_pattern, full_form, text, flags=re.IGNORECASE)
        return text
    
    def remove_filler_words(self, text: str) -> str:
        """Remove conversational filler words that don't add semantic value"""
        filler_pattern = r'\b(' + '|'.join(re.escape(f) for f in self.filler_words) + r')\b'
        text = re.sub(filler_pattern, '', text, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', text).strip()
    
    def normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation for consistency"""
        # Replace multiple punctuation with single
        text = re.sub(r'[?]+', '?', text)
        text = re.sub(r'[!]+', '!', text)
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[,]+', ',', text)
        
        # Remove trailing punctuation
        text = re.sub(r'[?.!,;:]+\s*$', '', text)
        
        return text
    
    def extract_temporal_references(self, text: str) -> Tuple[str, List[str]]:
        """
        Extract temporal references using keyword matching and patterns.
        Returns: (unchanged_text, temporal_entities)
        """
        temporal_entities = []
        text_lower = text.lower()
        
        # Keyword matching with word boundaries
        for keyword in self.temporal_keywords:
            if re.search(rf'\b{re.escape(keyword)}\b', text_lower):
                temporal_entities.append(keyword)
        
        # Date/time patterns with actual values
        date_patterns = [
            (r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 'date'),
            (r'\d{4}[/-]\d{1,2}[/-]\d{1,2}', 'date'),
            (r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b', 'time')
        ]
        
        for pattern, label in date_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                # Add actual matched values for debugging
                temporal_entities.extend([f"{label}:{m}" for m in matches])
        
        return text, list(set(temporal_entities))
    
    def extract_spatial_references(self, text: str) -> Tuple[str, List[str]]:
        """
        Extract spatial references using keyword matching and patterns.
        Returns: (unchanged_text, spatial_entities)
        """
        spatial_entities = []
        text_lower = text.lower()
        
        # Keyword matching with word boundaries
        for keyword in self.spatial_keywords:
            if re.search(rf'\b{re.escape(keyword)}\b', text_lower):
                spatial_entities.append(keyword)
        
        # Route number patterns with actual values
        route_patterns = [
            (r'\b[iI]-?\d{1,3}\b', 'interstate'),
            (r'\bUS-?\d{1,3}\b', 'us_highway'),
            (r'\broute\s+\d{1,3}\b', 'route'),
        ]
        
        for pattern, label in route_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Add actual matched values
                spatial_entities.extend([f"{label}:{m.lower()}" for m in matches])
        
        return text, list(set(spatial_entities))
    
    def preprocess(self, prompt: Optional[str], debug: bool = False) -> Dict[str, Any]:
        """
        Main preprocessing pipeline.
        
        Pipeline order:
        1. Normalize whitespace
        2. Expand abbreviations (before lowercasing)
        3. Extract entities (before aggressive cleaning)
        4. Lowercase
        5. Remove filler words
        6. Normalize punctuation
        7. Remove negated terms
        8. Final cleanup
        
        Args:
            prompt: Raw user input
            debug: Whether to print debug information
        
        Returns:
            dict with:
                - original: Original prompt
                - cleaned: Fully preprocessed prompt
                - temporal_entities: Extracted time references
                - spatial_entities: Extracted location references
                - removed_negations: Terms removed due to negation
                - processing_steps: List of (step_name, result) tuples
        """
        steps = []
        
        # Step 1: Normalize whitespace
        text = self.normalize_whitespace(prompt)
        steps.append(("normalize_whitespace", text))
        
        # Step 2: Expand abbreviations (before lowercasing to preserve context)
        text = self.expand_abbreviations(text)
        steps.append(("expand_abbreviations", text))
        
        # Step 3 & 4: Extract entities BEFORE aggressive cleaning
        text, temporal_entities = self.extract_temporal_references(text)
        steps.append(("extract_temporal", text))
        
        text, spatial_entities = self.extract_spatial_references(text)
        steps.append(("extract_spatial", text))
        
        # Step 5: Lowercase for semantic matching
        text = text.lower()
        steps.append(("lowercase", text))
        
        # Step 6: Remove filler words
        text = self.remove_filler_words(text)
        steps.append(("remove_filler_words", text))
        
        # Step 7: Normalize punctuation
        text = self.normalize_punctuation(text)
        steps.append(("normalize_punctuation", text))
        
        # Step 8: Remove negated terms (LAST to preserve context)
        text, removed_negations = self.remove_negated_terms(text)
        steps.append(("remove_negated_terms", text))
        
        # Step 9: Final whitespace normalization
        text = self.normalize_whitespace(text)
        
        result = {
            'original': prompt,
            'cleaned': text,
            'temporal_entities': temporal_entities,
            'spatial_entities': spatial_entities,
            'removed_negations': removed_negations,
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
            if result['removed_negations']:
                print(f"Removed negations: {result['removed_negations']}")
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


def preprocess_prompt(prompt: Optional[str], debug: bool = False) -> Dict[str, Any]:
    """Convenience function for preprocessing"""
    preprocessor = get_preprocessor()
    return preprocessor.preprocess(prompt, debug=debug)