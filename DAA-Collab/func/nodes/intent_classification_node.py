from func.state.agent_state import AgentState
from utils.ML.ml_based_intent_classification.intent_classification import user_query_intent_extraction
from utils.memory.episodic import get_intent_memory, store_intent_interaction
from utils.common import preprocess_prompt
from datetime import datetime
import uuid

def intent_classification(state: AgentState) -> AgentState:

    user_instruction = state.instruction
    
    # Generate execution ID if not present
    if not state.execution_id:
        state.execution_id = f"exec_{uuid.uuid4().hex[:12]}"
    
    print("\n" + "="*80)
    print("🎯 INTENT CLASSIFICATION NODE")
    print("="*80)
    print(f"Execution ID: {state.execution_id}")
    print(f"User Input: {user_instruction}\n")
    
    try:
        
        intent_threshold = 0.75  # Lower threshold, but always select at least 3
        min_intents = 3  # Always return at least 3 intents
        # Get preprocessed prompt for metadata
        preprocessed = preprocess_prompt(user_instruction, debug=False)

        # Perform intent classification
        intent_dict, confidence_scores, top_known_intents, unknown_intents = user_query_intent_extraction(
            prompt=preprocessed['cleaned'],
            threshold=intent_threshold,
            min_intents=min_intents,
            debug=True
        )
        
        # Store in LangGraph memory
        memory = get_intent_memory()
        checkpoint_id = memory.store_interaction(
            prompt=user_instruction,
            cleaned_prompt=preprocessed['cleaned'],
            all_intents=intent_dict['intents'],  # Combined known + unknown
            confidence_scores=confidence_scores,
            # THIS IS NOT NEEDED NOW
            # metadata={
            #     'temporal_entities': preprocessed.get('temporal_entities', []),
            #     'spatial_entities': preprocessed.get('spatial_entities', []),
            #     'threshold': intent_threshold,
            #     'known_intents': predicted_intents,
            #     'unknown_intents': unknown_intents,
            #     'node_name': 'intent_classification',
            #     'graph_execution_id': state.execution_id,
            #     'preprocessing_steps': preprocessed.get('processing_steps', [])
            # }
        )
        
        print(f"💾 Stored in LangGraph Memory")
        print(f"   Checkpoint ID: {checkpoint_id}")
        print(f"   Known Intents: {top_known_intents}")
        print(f"   Unknown Intents: {unknown_intents}")
        print(f"   Combined: {intent_dict['intents']}")
        
        print(f"\nTop 5 Confidence Scores (Known Intents):")
        for intent, score in sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
            status = "✓" if intent in top_known_intents else " "
            print(f"   [{status}] {intent:25s}: {score:.3f}")
        
        # Update state
        state.intent_classification = intent_dict
        state.mem_checkpoint_id = checkpoint_id
        state.predicted_classified_intents = intent_dict['intents']  # All intents
        state.known_intents = top_known_intents
        state.unknown_intents = unknown_intents
        state.intent_confidence_scores = confidence_scores
        state.classification_success = True
        state.needs_tool_generation = len(unknown_intents) > 0
        
        if unknown_intents:
            print(f"\n⚠️  Detected {len(unknown_intents)} unknown intent(s)")
            print(f"   These may require MCP tool generation or clarification")
            
            # Incrementally add unknown intents to centroids (background)
            try:
                from utils.ML.ml_based_intent_classification.incremental_centroid_update import add_intent_to_centroids
                import threading
                
                user_query = state.instruction
                for unknown_intent in unknown_intents:
                    # Run in background thread to not block response
                    thread = threading.Thread(
                        target=add_intent_to_centroids,
                        args=(unknown_intent, user_query),
                        daemon=True
                    )
                    thread.start()
                    print(f"   🔄 Adding '{unknown_intent}' to centroids (background)")
            except Exception as e:
                print(f"   ⚠️  Could not update centroids: {e}")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("   Please build the embeddings for the past user dataset before running the test.")
        state.classification_success = False
        state.error_log = [str(e)]
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        state.classification_success = False
        state.error_log = [str(e)]
    
    print("="*80 + "\n")
    
    return state