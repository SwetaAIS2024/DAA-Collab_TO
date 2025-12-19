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

def user_query_intent_extraction(
    prompt: str, 
    threshold: float = 0.7,
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
        debug=debug
    )
    
    all_intents = top_known_intents.copy()

    for unknown in unknown_intents:
        all_intents.append(unknown)

    intent_classification_result = {
        "intents": all_intents,
        "known_intents": top_known_intents,
        "unknown_intents": unknown_intents
    }


    return intent_classification_result, confidence_scores, top_known_intents, unknown_intents

def get_embedding_model(model_name: str = 'Qwen/Qwen3-Embedding-8B') -> SentenceTransformer:

    # Detect device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Load pre-trained sentence transformer (cache for offline use)
    try:
        # model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODEL_DIR) 
        embedding_model = SentenceTransformer(model_name, cache_folder = MODEL_DIR, trust_remote_code=True, device=DEVICE)
        print("✅ Sentence Transformer model loaded on device:", DEVICE)
    except Exception as e:
        print(f"⚠️ Error loading Sentence Transformer: {e}")
        print("   Run: pip install sentence-transformers")
        embedding_model = None
    
    if embedding_model is None:
        raise RuntimeError("Sentence Transformer not available. Install: pip install sentence-transformers")

    return embedding_model

def get_all_intents(
    prompt: str, 
    threshold: float = 0.7, 
    top_k: int = 3, # limit to top K
    debug: bool = True,
) -> Tuple[List[str], Dict[str, float], List[str]]:
    """
    Predict intents using semantic similarity + LLM extraction.
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
        # Ensure float for JSON serialization
        similarities[intent] = float(similarity)
    
    top_intent, top_score = max(similarities.items(), key=lambda x: x[1])

    if debug:
        print(f"\n{'='*80}")
        print("SIMILARITY SCORES")
        print(f"{'='*80}")
        for intent, score in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
            status = "✓" if score > threshold else " "
            print(f"  [{status}] {intent:25s}: {score:.3f}")
        print(f"{'='*80}\n")
    
    # FIX for getting too many intents
    # Sort by similarity
    sorted_intents = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # Apply BOTH threshold AND top-k filtering
    predicted = []
    for intent, score in sorted_intents[:top_k]:  # Only consider top K
        if score > threshold:
            predicted.append(intent)
    
    # Ensure at least we have a minimum relevance gap
    if len(predicted) > 1:
        # Check if there's a significant score drop
        scores = [similarities[i] for i in predicted]
        score_gap = scores[0] - scores[-1]
        
        if score_gap < 0.05:  # If all scores are too similar, keep only top 2
            predicted = predicted[:2]
            if debug:
                print(f"⚠️  Scores too similar (gap: {score_gap:.3f}), limiting to top 2\n")

    # ALWAYS run LLM extraction for unknown intents
    # This runs in parallel to known intent detection
    unknown_intents = extract_unknown_intents_llm(
        prompt=prompt,
        known_intents=predicted,
        known_intent_classes=intent_classes,
        debug=debug
    )
    
    # Fallback: if NO intents detected at all (known OR unknown), use top intent
    if not predicted and not unknown_intents:
        if top_score > 0.3:  # Minimum fallback threshold
            predicted = [top_intent]
            if debug:
                print(f"⚠️  Fallback: Using top intent '{top_intent}' (score: {top_score:.3f})\n")
    
    if debug:
        print(f"\n📊 FINAL RESULTS:")
        print(f"   Known intents: {predicted if predicted else 'None'}")
        print(f"   Unknown intents: {unknown_intents if unknown_intents else 'None'}")
        print(f"   Combined: {predicted + unknown_intents}\n")
    
    return predicted, similarities, unknown_intents

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
    
    # Detect device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    ext_llm_tokenizer, ext_llm_model = None, None
    MODEL_EXTRACT_UNKNOWN = 'Qwen/Qwen2.5-7B-Instruct' # this model is used for extracting the unknown intents
    
    try:
        if ext_llm_model is None or ext_llm_tokenizer is None:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            print(f"🔄 Loading extraction LLM model: {MODEL_EXTRACT_UNKNOWN}")
            ext_llm_tokenizer = AutoTokenizer.from_pretrained(
                MODEL_EXTRACT_UNKNOWN, 
                cache_dir=MODEL_DIR, 
                trust_remote_code=True
                )
            ext_llm_model = AutoModelForCausalLM.from_pretrained(
                MODEL_EXTRACT_UNKNOWN, 
                cache_dir=MODEL_DIR, 
                trust_remote_code=True, 
                dtype=torch.float16 if DEVICE=="cuda" else torch.float32,
                device_map="auto" if DEVICE=="cuda" else None,
                low_cpu_mem_usage=True
                )
            print("✅ Extraction LLM model loaded.")
    except Exception as e:
        print(f"⚠️ Error loading extraction LLM: {e}")
        ext_llm_tokenizer = None
        ext_llm_model = None
    return ext_llm_tokenizer, ext_llm_model

def extract_unknown_intents_llm(
    prompt: str,
    known_intents: List[str],
    known_intent_classes: List[str],
    debug: bool = False
) -> List[str]:

    import json
    # Detect device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    try:
        # Load local LLM
        ext_llm_tokenizer, ext_llm_model = load_unknown_intent_extractor_llm()
        
        # Create extraction prompt
        system_prompt = f"""You are an intent extraction expert for traffic analysis systems.

        Known intent types in the system:
        {', '.join(known_intent_classes)}

        Your task:
        1. Analyze the user prompt
        2. Identify ANY intents or tasks mentioned
        3. Compare against the known intent types
        4. Extract ONLY the intents that are NOT covered by the known types

        CRITICAL RULES:
        1. ONLY extract intents that are COMPLETELY DIFFERENT from the known types
        2. If the user's request can be satisfied by ANY combination of known intents, return []
        3. Extract ONLY if the request involves entirely new capabilities (e.g., "send email", "generate PDF")

        Return ONLY a JSON array of unknown intent descriptions (1-3 words each). If all intents are covered by known types, return an empty array [].

        Example:
        User: "Detect incidents and generate a PDF report"
        Known intents detected: ["incident_detection"]
        Output: ["report_generation", "pdf_export"]

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
        inputs = ext_llm_tokenizer([text], return_tensors="pt").to(DEVICE)
        
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
                extracted = eval(list_str)
                
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
