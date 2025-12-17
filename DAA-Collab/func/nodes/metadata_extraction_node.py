# from func.state.agent_state import AgentState

# def metadata_extraction(state: AgentState) -> AgentState:
   
#     return state


import pandas as pd
import os
from typing import Dict, Any, List
import json
import re
from PIL import Image
import PyPDF2
from func.state.agent_state import AgentState
from typing import Optional


# Import SHARED LLM from intent classification
from utils.ML.ml_based_intent_classification.model_transformer import ext_llm_tokenizer, ext_llm_model, MODEL_EXTRACT_UNKNOWN


def extract_file_content(file_path: str) -> Dict[str, Any]:
    """
    Extract raw content from any file type.
    
    Supports: CSV, Excel, JSON, Text, PDF, Images
    
    Returns:
        Dict with file info, preview, and structured data (if applicable)
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    content = {
        'file_path': file_path,
        'file_name': os.path.basename(file_path),
        'file_type': file_ext,
        'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
        'raw_content': None,
        'structured_data': None,
        'preview': None
    }
    
    try:
        # CSV files
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
            content['structured_data'] = df
            content['raw_content'] = df.to_string(max_rows=10, max_cols=10)
            content['preview'] = {
                'rows': len(df),
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'head': df.head(3).to_dict('records'),
                'sample_values': {
                    col: [str(v) for v in df[col].dropna().unique()[:3].tolist()]
                    for col in df.columns
                }
            }
        
        # Excel files
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            content['structured_data'] = df
            content['raw_content'] = df.to_string(max_rows=10)
            content['preview'] = {
                'rows': len(df),
                'columns': list(df.columns),
                'head': df.head(3).to_dict('records'),
                'sample_values': {
                    col: [str(v) for v in df[col].dropna().unique()[:3].tolist()]
                    for col in df.columns
                }
            }
        
        # JSON files
        elif file_ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            content['raw_content'] = json.dumps(data, indent=2)[:1500]
            content['preview'] = {
                'type': str(type(data).__name__),
                'keys': list(data.keys()) if isinstance(data, dict) else None,
                'length': len(data) if isinstance(data, (list, dict)) else None,
                'sample': str(data)[:500]
            }
        
        # Text files
        elif file_ext in ['.txt', '.log']:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            content['raw_content'] = text[:2000]
            content['preview'] = {
                'lines': len(text.split('\n')),
                'chars': len(text),
                'words': len(text.split()),
                'sample': text[:500]
            }
        
        # PDF files
        elif file_ext == '.pdf':
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages[:5]:
                    text += page.extract_text() + "\n"
            content['raw_content'] = text[:2000]
            content['preview'] = {
                'pages': len(reader.pages),
                'sample_text': text[:500]
            }
        
        # Image files
        elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
            img = Image.open(file_path)
            content['preview'] = {
                'width': img.width,
                'height': img.height,
                'mode': img.mode,
                'format': img.format
            }
            content['raw_content'] = f"Image: {img.width}x{img.height}, mode={img.mode}"
        
        else:
            content['raw_content'] = f"Unsupported file type: {file_ext}"
    
    except Exception as e:
        print(f"⚠️  Error reading file: {e}")
        content['raw_content'] = f"Error: {str(e)}"
    
    return content


def generate_metadata_with_llm(
    file_content: Dict[str, Any], 
    user_query: Optional[str],
    intents: List[str],
    debug: bool = True
) -> Dict[str, Any]:
    """
    Use SHARED LLM to intelligently extract metadata.
    
    Args:
        file_content: Extracted file content
        user_query: User's original query
        intents: Classified intents from previous node
        debug: Enable debug output
    
    Returns:
        Structured metadata with field suggestions, filters, etc.
    """
    global ext_llm_model, ext_llm_tokenizer
    
    if ext_llm_model is None or ext_llm_tokenizer is None:
        print("⚠️  LLM model not loaded, using fallback metadata")
        return generate_fallback_metadata(file_content, intents)
    
    # Build context-aware prompt
    system_prompt = """You are a data analysis assistant. Analyze the file and extract metadata relevant to the user's query and detected intents.

Tasks:
1. Identify columns/fields relevant to the query
2. Detect temporal, spatial, categorical, numeric fields
3. Suggest filters based on the query
4. Recommend analysis approaches
5. Flag data quality issues

Return ONLY valid JSON in this format:
{
  "data_summary": "Brief description",
  "relevant_fields": ["field1", "field2"],
  "temporal_fields": ["date", "time"],
  "spatial_fields": ["location", "lat", "lon"],
  "categorical_fields": ["type", "category"],
  "numeric_fields": ["count", "value"],
  "suggested_filters": {"field": "value"},
  "analysis_suggestions": ["suggestion1", "suggestion2"],
  "data_quality_issues": ["issue1"]
}
"""

    # Build file preview
    file_preview = f"""File: {file_content['file_name']}
Type: {file_content['file_type']}
Size: {file_content['file_size_mb']} MB
"""

    if file_content['preview']:
        if file_content['file_type'] in ['.csv', '.xlsx', '.xls']:
            preview = file_content['preview']
            file_preview += f"""
Rows: {preview.get('rows', 'N/A')}
Columns: {', '.join(preview.get('columns', []))}

Sample Data (first 3 rows):
{json.dumps(preview.get('head', [])[:3], indent=2)}

Sample Values:
{json.dumps(preview.get('sample_values', {}), indent=2)}
"""
        else:
            file_preview += f"\nPreview:\n{json.dumps(file_content['preview'], indent=2)}"
    
    if file_content['raw_content']:
        file_preview += f"\n\nContent Sample:\n{file_content['raw_content'][:1000]}"

    user_prompt = f"""User Query: "{user_query}"
Detected Intents: {', '.join(intents)}

File Information:
{file_preview}

Analyze and return metadata as JSON.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        if debug:
            print(f"🤖 Using shared LLM: {MODEL_EXTRACT_UNKNOWN}")
        
        text = ext_llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = ext_llm_tokenizer([text], return_tensors="pt").to(ext_llm_model.device)
        
        import torch
        with torch.no_grad():
            outputs = ext_llm_model.generate(
                **inputs,
                max_new_tokens=800,
                temperature=0.3,
                do_sample=True,
                top_p=0.9
            )
        
        response = ext_llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("assistant")[-1].strip()
        
        if debug:
            print(f"🤖 LLM Metadata Response:\n{response[:500]}...\n")
        
        # Parse JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            metadata = json.loads(json_match.group())
            return metadata
        else:
            if debug:
                print("⚠️  No valid JSON in response, using fallback")
            return generate_fallback_metadata(file_content, intents)
    
    except Exception as e:
        if debug:
            print(f"❌ LLM metadata extraction failed: {e}")
        return generate_fallback_metadata(file_content, intents)


def generate_fallback_metadata(file_content: Dict[str, Any], intents: List[str]) -> Dict[str, Any]:
    """
    Fallback metadata extraction using heuristics.
    """
    columns = []
    if file_content.get('preview') and file_content['file_type'] in ['.csv', '.xlsx', '.xls']:
        columns = file_content['preview'].get('columns', [])
    
    # Heuristic field detection
    temporal_keywords = ['date', 'time', 'timestamp', 'day', 'month', 'year', 'hour']
    spatial_keywords = ['location', 'lat', 'lon', 'latitude', 'longitude', 'highway', 'road', 'address', 'place']
    
    temporal_fields = [col for col in columns if any(kw in col.lower() for kw in temporal_keywords)]
    spatial_fields = [col for col in columns if any(kw in col.lower() for kw in spatial_keywords)]
    
    return {
        "data_summary": f"File contains {len(columns)} columns",
        "relevant_fields": columns[:5],
        "temporal_fields": temporal_fields,
        "spatial_fields": spatial_fields,
        "categorical_fields": [],
        "numeric_fields": [],
        "suggested_filters": {},
        "analysis_suggestions": [
            f"Explore {', '.join(intents)} analysis" if intents else "Perform exploratory analysis"
        ],
        "data_quality_issues": []
    }


def generate_natural_summary(metadata: Dict[str, Any]) -> str:
    """
    Convert metadata to natural language summary for planner context.
    """
    summary = f"""File: {metadata['file_info']['name']} ({metadata['file_info']['type']})
Size: {metadata['file_info']['size_mb']} MB

{metadata.get('data_summary', 'Data file loaded')}
"""
    
    if metadata.get('relevant_fields'):
        summary += f"\n\nKey Fields: {', '.join(metadata['relevant_fields'][:5])}"
    
    if metadata.get('temporal_fields'):
        summary += f"\nTemporal: {', '.join(metadata['temporal_fields'])}"
    
    if metadata.get('spatial_fields'):
        summary += f"\nSpatial: {', '.join(metadata['spatial_fields'])}"
    
    if metadata.get('numeric_fields'):
        summary += f"\nNumeric: {', '.join(metadata['numeric_fields'][:3])}"
    
    if metadata.get('suggested_filters'):
        summary += f"\n\nRecommended Filters:"
        for field, value in list(metadata['suggested_filters'].items())[:3]:
            summary += f"\n  • {field} = {value}"
    
    if metadata.get('analysis_suggestions'):
        summary += f"\n\nSuggested Analysis:"
        for suggestion in metadata['analysis_suggestions'][:3]:
            summary += f"\n  • {suggestion}"
    
    if metadata.get('data_quality_issues'):
        summary += f"\n\n⚠️  Data Quality:"
        for issue in metadata['data_quality_issues'][:3]:
            summary += f"\n  • {issue}"
    
    return summary


def metadata_extraction(state: AgentState) -> AgentState:
    """
    Main metadata extraction node.
    
    Flow:
    1. Extract file content
    2. Use LLM to generate intelligent metadata
    3. Enrich state with metadata for planner
    
    Args:
        state: Current graph state
            Required keys: file_path, user_input, classified_intents
    
    Returns:
        Updated state with file_metadata, file_content, dataframe
    """
    print(f"\n{'='*80}")
    print("📊 METADATA EXTRACTION NODE")
    print(f"{'='*80}")
    
    file_path = state.dataset_path
    user_query = state.instruction
    intents = state.predicted_classified_intents
    
    if not file_path:
        print("⚠️  No file provided, skipping metadata extraction")
        state.dataset_path = None
        # state.file_metadata = None
        # state.metadata_extracted = False
        return state 
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        state.dataset_path = None
        # state.file_metadata = None
        # state.metadata_extracted = False
        return state 
    
    print(f"File: {file_path}")
    print(f"Query: {user_query}")
    print(f"Intents: {', '.join(intents)}")
    
    try:
        # Step 1: Extract raw file content
        print("\n🔍 Step 1: Extracting file content...")
        file_content = extract_file_content(file_path)
        
        print(f"   ✓ Type: {file_content['file_type']}")
        print(f"   ✓ Size: {file_content['file_size_mb']} MB")
        
        if file_content['preview'] and 'columns' in file_content['preview']:
            print(f"   ✓ Columns: {len(file_content['preview']['columns'])}")
            print(f"   ✓ Rows: {file_content['preview'].get('rows', 'N/A')}")
        
        # Step 2: Generate intelligent metadata with LLM
        print("\n🤖 Step 2: Generating metadata with LLM...")
        metadata = generate_metadata_with_llm(
            file_content=file_content,
            user_query=user_query,
            intents=intents,
            debug=True
        )
        
        # Step 3: Add file info
        metadata['file_info'] = {
            'path': file_path,
            'name': file_content['file_name'],
            'type': file_content['file_type'],
            'size_mb': file_content['file_size_mb']
        }
        
        # Step 4: Generate natural language summary
        print("\n📝 Step 3: Generating summary...")
        summary = generate_natural_summary(metadata)
        metadata['summary'] = summary
        
        print(f"\n{'='*40}")
        print("METADATA SUMMARY")
        print(f"{'='*40}")
        print(summary)
        print(f"{'='*40}")
        
        print(f"\n✅ Metadata extraction complete")
        print(f"{'='*80}\n")
        
        # Return enriched state
        return {
            **state,
            'file_metadata': metadata,
            'file_content': file_content,
            'dataframe': file_content.get('structured_data'),  # For CSV/Excel
            'metadata_extracted': True
        }
    
    except Exception as e:
        print(f"❌ Error in metadata extraction: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            **state,
            'file_metadata': None,
            'metadata_extracted': False,
            'error': str(e)
        }