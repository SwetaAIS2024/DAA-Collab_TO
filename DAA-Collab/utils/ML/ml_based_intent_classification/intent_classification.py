import pandas as pd
from ast import literal_eval
import joblib
import os, sys
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.common import preprocess_prompt
import torch
# MODEL_NAME = 'Alibaba-NLP/gte-large-en-v1.5'

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(os.path.join(BASE_DIR, "DAA-Collab"))
DATA_PATH = os.path.join(BASE_DIR, "DAA-Collab/data/user_prompt_dataset/traffic_prompts_realistic_option2_FIXED.csv")
MODEL_DIR = os.path.join(BASE_DIR,"DAA-Collab/utils/ML/ml_based_intent_classification/saved_models_transformer")
os.makedirs(MODEL_DIR, exist_ok=True)

# global cache for the  models (to avoid the re-loading)
_embedding_model = None
_ext_llm_tokenizer = None
_ext_llm_model = None
_DEVICE = None


def user_query_intent_extraction(
    prompt: str, 
    threshold: float = 0.45,
    min_intents: int = 3,
    debug: bool = False
) -> Tuple[Dict[str, List[str]], Dict[str, float], List[str], List[str]]:

    # get the embeddings 
    # build_intent_centroids_index(include_memory_data=True) 
    # Cannot run this during user query time
    # Run the build_centroids_script.py to build the centroids index.

    # Get predicted intents and confidence scores
    top_known_intents, confidence_scores, unknown_intents = get_all_intents(
        prompt=prompt,
        threshold=threshold,
        min_intents=min_intents,
        debug=debug
    )
    
    all_intents = top_known_intents + unknown_intents

    intent_classification_result = {
        "intents": all_intents,
        "known_intents": top_known_intents,
        "unknown_intents": unknown_intents
    }

    return intent_classification_result, confidence_scores, top_known_intents, unknown_intents

def get_embedding_model(model_name: str = 'Qwen/Qwen3-Embedding-8B') -> SentenceTransformer:

    global _embedding_model, _DEVICE

    # directly return the cached model if available 
    if _embedding_model is not None:
        return _embedding_model

    # Detect device
    if _DEVICE is None:
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️ Using device: {_DEVICE}")
        if _DEVICE == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Load pre-trained sentence transformer (cache for offline use)
    try:
        # model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODEL_DIR) 
        _embedding_model = SentenceTransformer(
            model_name, 
            cache_folder = MODEL_DIR, 
            trust_remote_code=True, 
            device=_DEVICE
            )
        print("✅ Sentence Transformer model loaded on device:", _DEVICE)
    except Exception as e:
        print(f"⚠️ Error loading Sentence Transformer: {e}")
        raise RuntimeError("Sentence transformer model not available")
       
    return _embedding_model

def get_all_intents(
    prompt: str, 
    threshold: float = 0.45,  # FIXED: Lowered from 0.5
    top_k: int = 5,
    min_intents: int = 3,  # Always return at least this many intents
    debug: bool = True,
) -> Tuple[List[str], Dict[str, float], List[str]]:
    """
    Predict intents using semantic similarity + LLM extraction.
    Always returns at least min_intents, even if they are below threshold.
    """
    model = get_embedding_model()

    if model is None:
        raise RuntimeError("Sentence Transformer not loaded")
    
    # Load model
    intent_prototypes, intent_classes = load_past_intent_centroids_index()
       
    # Get prompt embedding
    prompt_embedding = model.encode([prompt], show_progress_bar=False)[0]
    
    # Calculate similarity to each intent prototype
    similarities = {}
    for intent in intent_classes:
        prototype = intent_prototypes[intent]
        similarity = cosine_similarity(
            np.array([prompt_embedding]), 
            np.array([prototype])
        )[0][0]
        similarities[intent] = float(similarity)
    
    top_intent, top_score = max(similarities.items(), key=lambda x: x[1])

    if debug:
        print(f"\n{'='*80}")
        print("SIMILARITY SCORES")
        print(f"{'='*80}")
        for intent, score in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
            status = "✓" if score >= threshold else " "
            print(f"  [{status}] {intent:25s}: {score:.3f}")
        print(f"{'='*80}\n")
    
    # Sort by similarity
    sorted_intents = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # Select top K intents above threshold, but ensure at least min_intents
    predicted = []
    for intent, score in sorted_intents[:top_k]:
        if score >= threshold:
            predicted.append(intent)
    
    # Ensure at least min_intents are returned (even if below threshold)
    if len(predicted) < min_intents:
        for intent, score in sorted_intents[:min_intents]:
            if intent not in predicted:
                predicted.append(intent)
        # Sort predicted to maintain score order
        predicted = [intent for intent, _ in sorted_intents if intent in predicted][:min_intents]
    
    # DEBUG: Show initial selection
    if debug:
        print(f"📌 TOP {top_k} PREDICTIONS (threshold={threshold}, min={min_intents}):")
        for i, (intent, score) in enumerate(sorted_intents[:top_k]):
            status = "✓" if intent in predicted else " "
            print(f"   [{status}] {i+1}. {intent}: {score:.3f}")
        print()
    
    # Calculate score gap between top predictions to detect ambiguity
    # If multiple intents have similar high scores, the query may need more analysis
    score_gap = 0.0
    if len(sorted_intents) >= 2:
        score_gap = sorted_intents[0][1] - sorted_intents[1][1]
    
    # Count how many intents are above a "close to top" threshold (within 0.1 of top)
    close_intents_count = sum(1 for _, score in sorted_intents[:top_k] if score >= top_score - 0.1)
    
    # Extract unknown intents if:
    # 1. No predictions above threshold, OR
    # 2. Top score is weak (< 0.7), OR  
    # 3. Multiple intents have similar high scores (ambiguous query - gap < 0.1 and 2+ close intents)
    # 4. Top score is not very confident (< 0.85) - gives LLM a chance to find specialized intents
    extract_unknown_intents_flag = (
        # not predicted or
        # top_score < 0.7 or
        # (score_gap < 0.1 and close_intents_count >= 2) or
        top_score < 0.9
    )
    
    if extract_unknown_intents_flag:
        print(f"🔍 Unknown extraction triggered: top_score={top_score:.3f}, gap={score_gap:.3f}, close_intents={close_intents_count}")
        unknown_intents = extract_unknown_intents_llm(
            prompt=prompt,
            known_intents=predicted,
            known_intent_classes=intent_classes,
            debug=debug
        )
    else:
        unknown_intents = []
        if debug:
            print(f"⏭️  Skipping unknown intent extraction (confident match: {top_score:.3f}, gap: {score_gap:.3f})\n")
    
    # Fallback
    if not predicted and not unknown_intents:
        if top_score > 0.4:
            predicted = [top_intent]
            if debug:
                print(f"⚠️  Fallback: Using top intent '{top_intent}' (score: {top_score:.3f})\n")
    
    if debug:
        print(f"\n📊 RESULTS:")
        print(f"   Predicted intents: {predicted if predicted else 'None'}")
        print(f"   Unknown intents: {unknown_intents if unknown_intents else 'None'}\n")
    
    return predicted, similarities, unknown_intents


# Define canonical intents (the 10 original well-defined intents)
CANONICAL_INTENTS = [
    'visualization', 'incident_detection', 'spatio_temporal',
    'meta_attributes', 'traffic_impact', 'incident_classification',
    'traffic_anomaly', 'causal_analysis', 'traffic_forecasting',
    'report_generation'
]

def load_past_intent_centroids_index() -> Tuple[Dict[str, np.ndarray], List[str]]:

    prototypes_path = os.path.join(MODEL_DIR, "intent_centroids.joblib")
    classes_path = os.path.join(MODEL_DIR, "intent_classes.joblib")
    
    if not os.path.exists(prototypes_path) or not os.path.exists(classes_path):
        raise FileNotFoundError(
            f"Model files not found in {MODEL_DIR}\n"
            "Train the model first: python -m utils.ML.ml_based_intent_classification.model_transformer"
        )
    
    intent_prototypes = joblib.load(prototypes_path)
    intent_classes = joblib.load(classes_path)
    
    return intent_prototypes, intent_classes

def load_unknown_intent_extractor_llm():
    global _ext_llm_tokenizer, _ext_llm_model, _DEVICE
    
    #directly return the cached model if available
    if _ext_llm_tokenizer is not None and _ext_llm_model is not None:
        return _ext_llm_tokenizer, _ext_llm_model
    
    # Detect device once
    if _DEVICE is None:
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️ Using device: {_DEVICE}")
        if _DEVICE == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

    MODEL_EXTRACT_UNKNOWN = 'Qwen/Qwen2.5-7B-Instruct' # this model is used for extracting the unknown intents
    
    try: 
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"🔄 Loading extraction LLM model: {MODEL_EXTRACT_UNKNOWN}")
        _ext_llm_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_EXTRACT_UNKNOWN, 
            cache_dir=MODEL_DIR, 
            trust_remote_code=True
            )
        _ext_llm_model = AutoModelForCausalLM.from_pretrained(
            MODEL_EXTRACT_UNKNOWN, 
            cache_dir=MODEL_DIR, 
            trust_remote_code=True, 
            dtype=torch.float16 if _DEVICE=="cuda" else torch.float32,
            device_map="auto" if _DEVICE=="cuda" else None,
            low_cpu_mem_usage=True
            )
        print("✅ Extraction LLM model loaded.")
    except Exception as e:
        print(f"⚠️ Error loading extraction LLM: {e}")
        _ext_llm_tokenizer = None
        _ext_llm_model = None
    
    return _ext_llm_tokenizer, _ext_llm_model

def extract_unknown_intents_llm(
    prompt: str,
    known_intents: List[str],
    known_intent_classes: List[str],
    debug: bool = False
) -> List[str]:

    import json
   
    try:
        # Load local LLM
        ext_llm_tokenizer, ext_llm_model = load_unknown_intent_extractor_llm()
        
        if ext_llm_tokenizer is None or ext_llm_model is None:
            if debug:
                print("⚠️  Extraction LLM model not available.")
            return []

        # Map common synonyms to known intents
        SYNONYM_MAP = {
            'graph': 'visualization',
            'plot': 'visualization',
            'chart': 'visualization',
            'visual': 'visualization',
            'accident': 'incident_detection',
            'crash': 'incident_detection',
            'collision': 'incident_detection',
            'time': 'spatio_temporal',
            'date': 'spatio_temporal',
            'location': 'spatio_temporal',
            'timing': 'spatio_temporal',
        }

        # Pre-filter prompt terms
        prompt_lower = prompt.lower()
        likely_known = any(
            synonym in prompt_lower 
            for synonym in SYNONYM_MAP.keys()
        )

        # If prompt contains known synonyms AND we have known_intents detected,
        # be even MORE conservative with LLM extraction
        if likely_known and known_intents:
            # Increase filtering threshold
            max_similarity = 0.7  # Instead of 0.85


        # Create extraction prompt
        system_prompt = f"""You are an intent extraction expert for traffic analysis systems.

        Known intent types in the system:
        {', '.join(known_intent_classes)}

        Your task:
        Known intent types ALREADY in the system (DO NOT extract these):
        - visualization: Creating graphs, charts, plots, visual representations
        - incident_detection: Finding, detecting, identifying accidents/incidents
        - spatio_temporal: Location-based analysis, time-based analysis, date filtering
        - meta_attributes: Data attributes, severity, type classification
        - traffic_impact: Traffic flow analysis, congestion, impact assessment
        - incident_classification: Categorizing incident types
        - traffic_anomaly: Detecting unusual patterns

        Full list: {', '.join(known_intent_classes)}

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

        user_prompt = f"""User prompt: "{prompt}"

        Known intents already detected: {known_intents if known_intents else "None"}

        Unknown intents (JSON array):"""

        # Format messages for Qwen
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Apply chat template
        text = ext_llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = ext_llm_tokenizer([text], return_tensors="pt").to(_DEVICE)
        
        # Generate
        with torch.no_grad():
            outputs = ext_llm_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.5,
                do_sample=True,
                top_p=0.9,
                pad_token_id=ext_llm_tokenizer.eos_token_id
            )
        
        # Decode response
        response = ext_llm_tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        if debug:
            print(f"\n🤖 LLM Response: {response}")
        
        # Parse response
        try:
            import re
            match = re.search(r'\[([^\]]*)\]', response)
            if match:
                list_str = '[' + match.group(1) + ']'
                # extracted = eval(list_str)
                extracted = literal_eval(list_str) # safer than eval
                
                # CRITICAL FIX: Filter out known intent class names
                filtered = []
                for intent in extracted:
                    intent_lower = intent.lower().strip()
                    
                    # Check if it matches ANY known intent class (case-insensitive)
                    is_known_class = any(
                        intent_lower == known_class.lower() 
                        for known_class in known_intent_classes
                    )
                    
                    if is_known_class:
                        if debug:
                            print(f"   ⚠️  Filtered '{intent}' - matches known class")
                        continue
                    
                    # Also check similarity to known class names
                    from difflib import SequenceMatcher
                    max_similarity = max(
                        SequenceMatcher(None, intent_lower, kc.lower()).ratio()
                        for kc in known_intent_classes
                    )
                    
                    if max_similarity > 0.85:  # 85% similarity threshold
                        if debug:
                            print(f"   ⚠️  Filtered '{intent}' - too similar to known class")
                        continue
                    
                    filtered.append(intent)
                
                if debug:
                    print(f"🔍 Extracted unknown intents: {filtered}")
                
                return filtered
            else:
                return []
                
        except Exception as e:
            if debug:
                print(f"⚠️  Could not parse LLM response: {e}")
            return []
    
    except Exception as e:
        if debug:
            print(f"❌ LLM extraction error: {e}")
        return []
