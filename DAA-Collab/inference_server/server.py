"""
GPU Inference Server for Intent Classification
Run this on a machine with GPU to handle model inference
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import uvicorn
import json
import os

app = FastAPI(title="Intent Classification Inference Server")

# Global model instances
embedding_model = None
llm_tokenizer = None
llm_model = None
DEVICE = None

@app.on_event("startup")
async def load_models():
    """Load models once at server startup"""
    global embedding_model, llm_tokenizer, llm_model, DEVICE
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models on device: {DEVICE}")
    
    try:
        # Load embedding model with 8-bit quantization
        print("Loading embedding model: Qwen/Qwen3-Embedding-8B...")
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        
        embedding_model = SentenceTransformer(
            'Qwen/Qwen3-Embedding-8B',
            device=DEVICE,
            trust_remote_code=True,
            model_kwargs={
                'quantization_config': quant_config,
                'device_map': {'': 0}
            } if DEVICE == "cuda" else {}
        )
        print("Embedding model loaded successfully")
        
        # Load LLM with 4-bit quantization
        print("Loading LLM: Qwen/Qwen2.5-7B-Instruct...")
        llm_quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        llm_tokenizer = AutoTokenizer.from_pretrained(
            'Qwen/Qwen2.5-7B-Instruct',
            trust_remote_code=True
        )
        
        llm_model = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen2.5-7B-Instruct',
            trust_remote_code=True,
            quantization_config=llm_quant if DEVICE == "cuda" else None,
            device_map="auto" if DEVICE == "cuda" else None,
            low_cpu_mem_usage=True
        )
        print("LLM loaded successfully")
        
        if DEVICE == "cuda":
            print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB allocated")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        raise


# Request/Response Models
class EmbedRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

class ExtractRequest(BaseModel):
    query: str
    known_intents: List[str]
    known_classes: List[str]

class ExtractResponse(BaseModel):
    unknown_intents: List[str]


# API Endpoints
@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """Generate embeddings for given texts"""
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    
    try:
        embeddings = embedding_model.encode(
            request.texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return {"embeddings": embeddings.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-intents", response_model=ExtractResponse)
async def extract_intents(request: ExtractRequest):
    """Extract unknown intents using LLM"""
    if llm_tokenizer is None or llm_model is None:
        raise HTTPException(status_code=503, detail="LLM model not loaded")
    
    try:
        # Synonym mapping
        SYNONYM_MAP = {
            'graph': 'visualization',
            'plot': 'visualization',
            'chart': 'visualization',
            'display': 'visualization',
            'show': 'visualization',
            'email': 'email_notification',
            'notify': 'email_notification',
            'send': 'email_notification',
            'alert': 'email_notification'
        }
        
        # Build prompt
        prompt = f"""You are analyzing a traffic management query to find NEW intents not covered by existing ones.

Query: "{request.query}"

Known intents already available:
{chr(10).join(f"- {intent}" for intent in request.known_intents)}

Task: Identify ANY additional intents present in the query that are NOT already covered.

Rules:
1. Only return intents that are CLEARLY different from known ones
2. Use specific, descriptive names (e.g., "construction_impact" not "impact")
3. Return ONLY a JSON array of intent names
4. If no new intents found, return []

Response (JSON array only):"""

        messages = [{"role": "user", "content": prompt}]
        text = llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = llm_tokenizer(text, return_tensors="pt").to(llm_model.device)
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.1,
            top_p=0.9
        )
        
        response = llm_tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )
        
        # Parse JSON response
        try:
            response_stripped = response.strip()
            if response_stripped.startswith('```'):
                lines = response_stripped.split('\n')
                response_stripped = '\n'.join(
                    line for line in lines 
                    if not line.strip().startswith('```')
                )
            
            unknown_intents = json.loads(response_stripped)
            
            if isinstance(unknown_intents, list):
                # Filter and normalize
                filtered = []
                for intent in unknown_intents:
                    if not isinstance(intent, str) or not intent.strip():
                        continue
                    
                    intent = intent.strip().lower().replace(' ', '_')
                    
                    # Apply synonym mapping
                    intent = SYNONYM_MAP.get(intent, intent)
                    
                    # Check if matches known class
                    intent_lower = intent.lower()
                    is_known_class = any(
                        intent_lower == kc.lower() or 
                        intent_lower in kc.lower() or 
                        kc.lower() in intent_lower
                        for kc in request.known_classes
                    )
                    
                    if is_known_class:
                        continue
                    
                    # Check similarity to known classes
                    from difflib import SequenceMatcher
                    max_similarity = max(
                        SequenceMatcher(None, intent_lower, kc.lower()).ratio()
                        for kc in request.known_classes
                    )
                    
                    if max_similarity > 0.85:
                        continue
                    
                    filtered.append(intent)
                
                return {"unknown_intents": filtered}
            else:
                return {"unknown_intents": []}
                
        except Exception:
            return {"unknown_intents": []}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "embedding_model_loaded": embedding_model is not None,
        "llm_model_loaded": llm_model is not None
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Intent Classification Inference Server",
        "endpoints": {
            "POST /embed": "Generate text embeddings",
            "POST /extract-intents": "Extract unknown intents",
            "GET /health": "Health check"
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Intent Classification Inference Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()
    
    print(f"\nStarting server on {args.host}:{args.port}")
    print("Models will be loaded on startup...\n")
    
    uvicorn.run(app, host=args.host, port=args.port)
