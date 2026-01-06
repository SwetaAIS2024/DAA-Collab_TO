from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import os
import pandas as pd
from pathlib import Path
import threading
# FIX: Correct import for MemorySaver
from langgraph.checkpoint.memory import MemorySaver 

class IntentMemoryManager:
    """
    Episodic Memory Manager for intent classification tracking.
    
    Features:
    - Stores all user interactions with predicted intents
    - Tracks confidence scores for continuous learning
    - Supports user feedback/validation
    - Exports validated data for model retraining
    - Thread-safe JSONL storage for fast querying
    - Compatible with LangGraph agent state via checkpoint_id
    """
    
    def __init__(
            self, 
            checkpoint_dir: Optional[str] = None,
            export_dir: Optional[str] = None 
        ):
            # FIX: Resolve paths dynamically relative to the DAA-Collab root
            # Go up 2 levels: utils/memory -> utils -> DAA-Collab
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            
            # Use provided paths or defaults
            self.checkpoint_dir = checkpoint_dir or os.path.join(base_dir, "data/memory/episodic/checkpoints")
            self.export_dir = export_dir or os.path.join(base_dir, "data/memory/episodic/exports")
            
            print(f"DEBUG: Initializing IntentMemoryManager")
            print(f"DEBUG: Checkpoint Dir: {self.checkpoint_dir}")
            
            # Create directories if they don't exist
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            Path(self.export_dir).mkdir(parents=True, exist_ok=True)
            
            # Initialize metadata file path
            self.metadata_file = os.path.join(self.checkpoint_dir, "interactions_history.jsonl")
            print(f"episodic_memory: Storing interactions in {self.metadata_file}")
            
            # Initialize lock for thread safety
            self._lock = threading.Lock()
            
            # Initialize LangGraph MemorySaver
            self.memory = MemorySaver()
    
    def store_interaction(
        self,
        prompt: Optional[str],
        cleaned_prompt: Optional[str],
        all_intents: List[str],
        confidence_scores: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
        user_feedback: Optional[List[str]] = None
    ) -> str:
        """
        Store an intent classification interaction in episodic memory.
        
        Args:
            prompt: Original user prompt
            cleaned_prompt: Preprocessed prompt
            all_intents: List of all intent labels extracted from the user query
            confidence_scores: Dict[intent_name, similarity_score]
            metadata: Additional context (temporal/spatial entities, etc.)
            user_feedback: Optional corrected intents if available
        
        Returns:
            checkpoint_id: Unique identifier for this interaction
        """
        # Generate unique checkpoint ID
        checkpoint_id = f"intent_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Prepare state for storage
        state = {
            "checkpoint_id": checkpoint_id,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "cleaned_prompt": cleaned_prompt,
            "all_intents": all_intents,
            "confidence_scores": confidence_scores,
            "metadata": metadata or {},
            "user_feedback": user_feedback,
            "validated": user_feedback is not None,
            "feedback_timestamp": None,
            "feedback_notes": None
        }
        
        # Store metadata in JSONL (thread-safe)
        self._append_metadata(state)
        
        return checkpoint_id
    
    def _append_metadata(self, state: Dict[str, Any]) -> None:
        """Append interaction metadata to JSONL file (thread-safe, append-only for performance)"""
        try:
            with self._lock:
                # Ensure directory exists just in case
                os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
                
                print(f"DEBUG: Writing interaction {state.get('checkpoint_id')} to {self.metadata_file}")
                with open(self.metadata_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(state, ensure_ascii=False) + '\n')
                    f.flush() # Force write to disk
                    os.fsync(f.fileno()) # Ensure it hits the OS
                print(f"DEBUG: Write successful")
        except Exception as e:
            print(f"⚠️ Warning: Could not append to metadata file: {e}")
            import traceback
            traceback.print_exc()
    
    def add_feedback(
        self,
        checkpoint_id: str,
        correct_intents: List[str],
        feedback_notes: Optional[str] = None
    ) -> bool:
        """
        Add user feedback/correction to a stored interaction.
        
        Args:
            checkpoint_id: ID of the interaction to update
            correct_intents: User-corrected intent labels
            feedback_notes: Optional notes about the correction
        
        Returns:
            bool: True if feedback was successfully added
        """
        try:
            # Load all interactions
            interactions: List[Dict[str, Any]] = []
            state_to_update: Optional[Dict[str, Any]] = None
            
            with self._lock:
                if os.path.exists(self.metadata_file):
                    with open(self.metadata_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                state = json.loads(line.strip())
                                if state.get('checkpoint_id') == checkpoint_id:
                                    # Store original predictions before overwriting
                                    state['original_predictions'] = state.get('all_intents', [])
                                    # Update all_intents with user-corrected intents
                                    state['all_intents'] = correct_intents
                                    # Also store in user_feedback for tracking
                                    state['user_feedback'] = correct_intents
                                    state['feedback_notes'] = feedback_notes
                                    state['feedback_timestamp'] = datetime.now().isoformat()
                                    state['validated'] = True
                                    state_to_update = state
                                interactions.append(state)
                            except json.JSONDecodeError:
                                continue
                
                if state_to_update is None:
                    print(f"⚠️ Checkpoint {checkpoint_id} not found in memory")
                    return False
                
                # Write back all interactions
                with open(self.metadata_file, 'w', encoding='utf-8') as f:
                    for interaction in interactions:
                        f.write(json.dumps(interaction, ensure_ascii=False) + '\n')
            
            return True
                
        except Exception as e:
            print(f"⚠️ Error adding feedback: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_recent_interactions(
        self, 
        n: int = 10, 
        validated_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get N most recent interactions.
        
        Args:
            n: Number of interactions to retrieve
            validated_only: Only return interactions with user feedback
        
        Returns:
            List of interaction states (most recent first)
        """
        interactions: List[Dict[str, Any]] = []
        
        if not os.path.exists(self.metadata_file):
            return interactions
        
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        state = json.loads(line.strip())
                        if validated_only and not state.get('validated', False):
                            continue
                        interactions.append(state)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"⚠️ Error reading metadata file: {e}")
        
        # Sort by timestamp (most recent first)
        interactions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return interactions[:n]
    
    def get_interactions_by_intent(self, intent: str) -> List[Dict[str, Any]]:
        """
        Get all interactions that predicted a specific intent.
        
        Args:
            intent: Intent label to filter by
        
        Returns:
            List of interactions that predicted this intent
        """
        interactions: List[Dict[str, Any]] = []
        
        if not os.path.exists(self.metadata_file):
            return interactions
        
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        state = json.loads(line.strip())
                        if intent in state.get('all_intents', []):
                            interactions.append(state)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"⚠️ Error reading metadata file: {e}")
        
        return interactions
    
    def export_to_training_dataset(
        self, 
        validated_only: bool = True,
        output_file: Optional[str] = None
    ) -> Optional[str]:
        """
        Export interactions to CSV format for model retraining.
        
        Args:
            validated_only: Only export validated interactions with feedback
            output_file: Custom output file path (auto-generated if None)
        
        Returns:
            str: Path to exported CSV file, or None if no data
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(self.export_dir, f"retraining_dataset_{timestamp}.csv")
        
        # Get interactions
        # FIX: If validated_only is True, we might filter out everything if nothing is validated yet.
        # For testing/debugging, we might want to see unvalidated ones too if requested.
        # But the training script specifically asks for validated ones.
        # Let's add a debug print to see what's happening.
        interactions = self.get_recent_interactions(n=100000, validated_only=validated_only)
        
        print(f"DEBUG: Found {len(interactions)} interactions (validated_only={validated_only})")
        
        if not interactions:
            print("⚠️ No interactions to export")
            return None
        
        # Convert to training format
        training_data: List[Dict[str, Any]] = []
        for interaction in interactions:
            # Use corrected intents if available, otherwise predicted
            # FIX: Handle case where user_feedback is explicitly None (key exists but value is None)
            intents = interaction.get('user_feedback')
            if not intents:
                intents = interaction.get('all_intents', [])
            
            # Ensure intents is a list of strings, not None or empty if possible
            if intents is None:
                intents = []
            
            # Calculate average confidence
            confidence_scores = interaction.get('confidence_scores', {})
            avg_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.0
            
            training_data.append({
                'id': interaction.get('checkpoint_id', ''),
                'prompt': interaction.get('prompt', ''),
                'intents': str(intents),  # Format as string list like original dataset
                'timestamp': interaction.get('timestamp', ''),
                'confidence_avg': avg_confidence,
                'validated': interaction.get('validated', False),
                'feedback_notes': interaction.get('feedback_notes', '')
            })
        
        # Save to CSV
        try:
            df = pd.DataFrame(training_data)
            df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"✅ Exported {len(training_data)} interactions to {output_file}")
            return output_file
        except Exception as e:
            print(f"❌ Error exporting to CSV: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.
        
        Returns:
            Dict with statistics about stored interactions
        """
        all_interactions = self.get_recent_interactions(n=100000, validated_only=False)
        validated = [i for i in all_interactions if i.get('validated', False)]
        
        # Intent distribution
        intent_distribution: Dict[str, int] = {}
        for interaction in all_interactions:
            for intent in interaction.get('all_intents', []):
                intent_distribution[intent] = intent_distribution.get(intent, 0) + 1
        
        # Average confidence by intent
        intent_confidences: Dict[str, float] = {}
        intent_counts: Dict[str, int] = {}
        for interaction in all_interactions:
            for intent, score in interaction.get('confidence_scores', {}).items():
                if intent not in intent_confidences:
                    intent_confidences[intent] = 0.0
                    intent_counts[intent] = 0
                intent_confidences[intent] += score
                intent_counts[intent] += 1
        
        avg_confidence_by_intent = {
            intent: intent_confidences[intent] / intent_counts[intent]
            for intent in intent_confidences
            if intent_counts[intent] > 0
        }
        
        return {
            'total_interactions': len(all_interactions),
            'validated_interactions': len(validated),
            'validation_rate': len(validated) / len(all_interactions) if all_interactions else 0.0,
            'intent_distribution': intent_distribution,
            'avg_confidence_by_intent': avg_confidence_by_intent,
            'oldest_interaction': all_interactions[-1].get('timestamp') if all_interactions else None,
            'newest_interaction': all_interactions[0].get('timestamp') if all_interactions else None,
            'checkpoint_dir': self.checkpoint_dir,
            'export_dir': self.export_dir,
            'metadata_file_size': os.path.getsize(self.metadata_file) if os.path.exists(self.metadata_file) else 0
        }
    
    def clear_memory(self, keep_validated: bool = True) -> None:
        """
        Clear memory checkpoints.
        
        Args:
            keep_validated: If True, keep validated interactions; if False, clear everything
        """
        if keep_validated:
            # Keep only validated interactions
            validated = self.get_recent_interactions(n=100000, validated_only=True)
            
            try:
                with open(self.metadata_file, 'w', encoding='utf-8') as f:
                    for interaction in validated:
                        f.write(json.dumps(interaction, ensure_ascii=False) + '\n')
                
                print(f"✅ Cleared memory, kept {len(validated)} validated interactions")
            except Exception as e:
                print(f"❌ Error clearing memory: {e}")
        else:
            # Clear everything
            try:
                if os.path.exists(self.metadata_file):
                    os.remove(self.metadata_file)
                    Path(self.metadata_file).touch()
                print("✅ Cleared all memory")
            except Exception as e:
                print(f"❌ Error clearing memory: {e}")
    
    def get_low_confidence_interactions(self, threshold: float = 0.4, n: int = 20) -> List[Dict[str, Any]]:
        """
        Get interactions with low confidence scores for review.
        
        Args:
            threshold: Confidence threshold (interactions below this are returned)
            n: Maximum number of interactions to return
        
        Returns:
            List of low-confidence interactions
        """
        all_interactions = self.get_recent_interactions(n=100000, validated_only=False)
        
        low_confidence: List[Dict[str, Any]] = []
        for interaction in all_interactions:
            confidence_scores = interaction.get('confidence_scores', {})
            if not confidence_scores:
                continue
            
            # Get max confidence for this interaction
            max_confidence = max(confidence_scores.values())
            
            if (max_confidence < threshold):
                interaction['max_confidence'] = max_confidence
                low_confidence.append(interaction)
        
        # Sort by confidence (lowest first)
        low_confidence.sort(key=lambda x: x.get('max_confidence', 0.0))
        
        return low_confidence[:n]


# Singleton instance
_intent_memory: Optional[IntentMemoryManager] = None

def get_intent_memory() -> IntentMemoryManager:
    """
    Get or create singleton intent memory manager.
    
    Returns:
        IntentMemoryManager: Global memory manager instance
    """
    global _intent_memory
    if (_intent_memory is None):
        _intent_memory = IntentMemoryManager()
    return _intent_memory


# Convenience function for direct access
def store_intent_interaction(
    prompt: str,
    cleaned_prompt: str,
    all_intents: List[str],
    confidence_scores: Dict[str, float],
    **kwargs: Any
) -> str:
    """
    Convenience function to store an interaction directly.
    
    Returns:
        checkpoint_id: Unique identifier
    """
    memory = get_intent_memory()
    return memory.store_interaction(
        prompt=prompt,
        cleaned_prompt=cleaned_prompt,
        all_intents=all_intents,
        confidence_scores=confidence_scores,
        **kwargs
    )