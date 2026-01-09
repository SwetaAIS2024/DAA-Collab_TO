"""
Chainlit Web Interface for Intent Classification Node Testing
Allows team members to test intent classification through a chat interface
With interactive intent selection/correction flow
"""

import chainlit as cl
from chainlit.data.base import BaseDataLayer
from chainlit.types import Feedback, PaginatedResponse, PageInfo, Pagination, ThreadDict, ThreadFilter
from chainlit.element import Element, ElementDict
from chainlit.step import StepDict
from chainlit.user import PersistedUser, User
from func.state.agent_state import AgentState
from func.nodes.intent_classification_node import intent_classification
from utils.memory.episodic import get_intent_memory
import time
import json
import asyncio
import gc
import os
import uuid
from typing import Optional, Dict, List
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Directory for persisting chat history
CHAT_HISTORY_DIR = os.path.join(os.path.dirname(__file__), "data", "chat_history")
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)


class FileDataLayer(BaseDataLayer):
    """File-based data layer for Chainlit chat persistence"""
    
    def __init__(self):
        self.users_dir = os.path.join(CHAT_HISTORY_DIR, "users")
        self.threads_dir = os.path.join(CHAT_HISTORY_DIR, "threads")
        self.steps_dir = os.path.join(CHAT_HISTORY_DIR, "steps")
        self.elements_dir = os.path.join(CHAT_HISTORY_DIR, "elements")
        self.feedbacks_dir = os.path.join(CHAT_HISTORY_DIR, "feedbacks")
        
        # Create directories
        for dir_path in [self.users_dir, self.threads_dir, self.steps_dir, 
                         self.elements_dir, self.feedbacks_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def _read_json(self, filepath: str) -> Optional[Dict]:
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None
    
    def _write_json(self, filepath: str, data: Dict):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        user_file = os.path.join(self.users_dir, f"{identifier}.json")
        data = self._read_json(user_file)
        if data:
            return PersistedUser(
                id=data.get("id", identifier),
                identifier=data.get("identifier", identifier),
                createdAt=data.get("createdAt", datetime.now().isoformat()),
                metadata=data.get("metadata", {})
            )
        return None
    
    async def create_user(self, user: User) -> Optional[PersistedUser]:
        persisted = PersistedUser(
            id=user.identifier,
            identifier=user.identifier,
            createdAt=datetime.now().isoformat(),
            metadata=user.metadata or {}
        )
        user_file = os.path.join(self.users_dir, f"{user.identifier}.json")
        self._write_json(user_file, {
            "id": persisted.id,
            "identifier": persisted.identifier,
            "createdAt": persisted.createdAt,
            "metadata": persisted.metadata
        })
        return persisted
    
    async def delete_feedback(self, feedback_id: str) -> bool:
        feedback_file = os.path.join(self.feedbacks_dir, f"{feedback_id}.json")
        if os.path.exists(feedback_file):
            os.remove(feedback_file)
            return True
        return False
    
    async def upsert_feedback(self, feedback: Feedback) -> str:
        feedback_id = feedback.get("id") or str(uuid.uuid4())
        feedback_data = dict(feedback)
        feedback_data["id"] = feedback_id
        feedback_file = os.path.join(self.feedbacks_dir, f"{feedback_id}.json")
        self._write_json(feedback_file, feedback_data)
        return feedback_id
    
    async def create_element(self, element: Element):
        element_dict = element.to_dict()
        element_id = element_dict.get("id", str(uuid.uuid4()))
        element_file = os.path.join(self.elements_dir, f"{element_id}.json")
        self._write_json(element_file, element_dict)
    
    async def get_element(self, thread_id: str, element_id: str) -> Optional[ElementDict]:
        element_file = os.path.join(self.elements_dir, f"{element_id}.json")
        return self._read_json(element_file)
    
    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):
        element_file = os.path.join(self.elements_dir, f"{element_id}.json")
        if os.path.exists(element_file):
            os.remove(element_file)
    
    async def create_step(self, step_dict: StepDict):
        thread_id = step_dict.get("threadId")
        if not thread_id:
            return
        
        steps_file = os.path.join(self.steps_dir, f"{thread_id}.json")
        steps = self._read_json(steps_file) or []
        
        # Add or update step
        step_id = step_dict.get("id")
        existing_idx = next((i for i, s in enumerate(steps) if s.get("id") == step_id), None)
        if existing_idx is not None:
            steps[existing_idx] = dict(step_dict)
        else:
            steps.append(dict(step_dict))
        
        self._write_json(steps_file, steps)
    
    async def update_step(self, step_dict: StepDict):
        await self.create_step(step_dict)
    
    async def delete_step(self, step_id: str):
        # Would need to search through all step files
        for filename in os.listdir(self.steps_dir):
            if filename.endswith(".json"):
                steps_file = os.path.join(self.steps_dir, filename)
                steps = self._read_json(steps_file) or []
                steps = [s for s in steps if s.get("id") != step_id]
                self._write_json(steps_file, steps)
    
    async def get_thread_author(self, thread_id: str) -> str:
        thread = await self.get_thread(thread_id)
        if thread:
            return thread.get("userId", "")
        return ""
    
    async def delete_thread(self, thread_id: str):
        thread_file = os.path.join(self.threads_dir, f"{thread_id}.json")
        if os.path.exists(thread_file):
            os.remove(thread_file)
        steps_file = os.path.join(self.steps_dir, f"{thread_id}.json")
        if os.path.exists(steps_file):
            os.remove(steps_file)
    
    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        threads = []
        
        for filename in os.listdir(self.threads_dir):
            if filename.endswith(".json"):
                thread_file = os.path.join(self.threads_dir, filename)
                thread = self._read_json(thread_file)
                if thread:
                    # Apply user filter
                    if filters and filters.userId:
                        if thread.get("userId") == filters.userId:
                            threads.append(thread)
                    else:
                        threads.append(thread)
        
        # Sort by createdAt descending
        threads.sort(key=lambda x: x.get("createdAt", ""), reverse=True)
        
        # Apply pagination
        first = pagination.first if pagination.first else 20
        threads = threads[:first]
        
        has_next = len(threads) == first
        
        page_info = PageInfo(
            hasNextPage=has_next,
            startCursor=threads[0].get("id") if threads else None,
            endCursor=threads[-1].get("id") if threads else None
        )
        
        return PaginatedResponse(pageInfo=page_info, data=threads)
    
    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        thread_file = os.path.join(self.threads_dir, f"{thread_id}.json")
        thread = self._read_json(thread_file)
        if thread:
            # Load steps
            steps_file = os.path.join(self.steps_dir, f"{thread_id}.json")
            steps = self._read_json(steps_file) or []
            thread["steps"] = steps
            # Ensure userIdentifier is set (Chainlit expects this field)
            if "userId" in thread and "userIdentifier" not in thread:
                thread["userIdentifier"] = thread["userId"]
        return thread
    
    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        thread_file = os.path.join(self.threads_dir, f"{thread_id}.json")
        thread = self._read_json(thread_file) or {
            "id": thread_id,
            "createdAt": datetime.now().isoformat()
        }
        
        if name is not None:
            thread["name"] = name
        if user_id is not None:
            thread["userId"] = user_id
            thread["userIdentifier"] = user_id  # Chainlit expects this field
        if metadata is not None:
            thread["metadata"] = metadata
        if tags is not None:
            thread["tags"] = tags
        
        thread["updatedAt"] = datetime.now().isoformat()
        self._write_json(thread_file, thread)
    
    async def build_debug_url(self) -> str:
        return ""
    
    async def close(self) -> None:
        pass


# Use the proper decorator to register the data layer
@cl.data_layer
def get_file_data_layer():
    return FileDataLayer()


# Header-based authentication for chat history persistence
@cl.header_auth_callback
async def header_auth_callback(headers: dict) -> Optional[cl.User]:
    """
    Create an anonymous user for chat history persistence.
    This enables Chainlit to restore conversations on page refresh.
    """
    # Use a cookie-based identifier or create a new anonymous user
    user_agent = headers.get("user-agent", "unknown")
    # Create a consistent identifier based on available headers
    user_id = f"anonymous_{hash(user_agent) % 100000:05d}"
    
    return cl.User(identifier=user_id, metadata={"role": "tester"})


# All known canonical intents
CANONICAL_INTENTS = [
    'visualization', 'incident_detection', 'spatio_temporal',
    'meta_attributes', 'traffic_impact', 'incident_classification',
    'traffic_anomaly', 'causal_analysis', 'traffic_forecasting',
    'report_generation'
]


@cl.on_stop
async def on_stop():
    """Called when server stops (e.g., during auto-reload with -w flag)"""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()


@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    await cl.Message(
        content="👋 **Welcome to Intent Classification Testing Tool!**\n\n"
                "I'll help you test the intent classification system.\n\n"
                "**How it works:**\n"
                "1. Type any traffic-related query\n"
                "2. I'll show you the **Top 5 predicted intents**\n"
                "3. You can **select/deselect** the correct intents\n"
                "4. You can **add unknown intents** if needed\n\n"
                "**Example queries:**\n"
                "- *Show me accidents on Highway 1 last week*\n"
                "- *Generate a report of traffic violations*\n"
                "- *Why are there crashes at junction 5?*\n\n"
                "Try it now! 👇"
    ).send()
    
    cl.user_session.set("query_count", 0)
    cl.user_session.set("test_results", [])
    cl.user_session.set("pending_classification", None)


@cl.on_chat_resume
async def on_chat_resume(thread: dict):
    """Restore session when user resumes a previous chat thread"""
    # Restore session data from thread metadata
    metadata = thread.get("metadata", {})
    
    cl.user_session.set("query_count", metadata.get("query_count", 0))
    cl.user_session.set("test_results", metadata.get("test_results", []))
    cl.user_session.set("pending_classification", metadata.get("pending_classification"))
    
    # Session restored silently - no resume message needed


@cl.on_message
async def main(message: cl.Message):
    """Process user query and classify intent"""
    
    user_query = message.content
    
    # Update session stats
    query_count = cl.user_session.get("query_count", 0) + 1
    cl.user_session.set("query_count", query_count)
    
    # Create processing message
    processing_msg = cl.Message(content="🔄 Processing...")
    await processing_msg.send()
    
    start_time = time.time()
    
    # Run classification
    try:
        state = AgentState(instruction=user_query)
        result = await asyncio.to_thread(intent_classification, state)
        
        if not result.classification_success:
            error_msg = "\n".join(result.error_log) if result.error_log else "Unknown error"
            await processing_msg.remove()
            await cl.Message(content=f"❌ **Classification Failed:** {error_msg}").send()
            return
        
        confidence_scores = result.intent_confidence_scores
        known_intents = result.known_intents
        unknown_intents = result.unknown_intents
        checkpoint_id = result.mem_checkpoint_id
        
    except Exception as e:
        await processing_msg.remove()
        await cl.Message(content=f"❌ **Error:** {str(e)}").send()
        return
    
    processing_time = time.time() - start_time
    await processing_msg.remove()
    
    # Store pending classification for user validation
    # Include ALL predicted intents (canonical + non-canonical) in selected_intents
    cl.user_session.set("pending_classification", {
        "query": user_query,
        "confidence_scores": confidence_scores,
        "known_intents": list(known_intents) if known_intents else [],
        "unknown_intents": list(unknown_intents) if unknown_intents else [],
        "selected_intents": list(known_intents) if known_intents else [],  # ALL predicted intents
        "custom_unknown_intents": [],
        "checkpoint_id": checkpoint_id,
        "processing_time": processing_time
    })
    
    # Build response with Top 5 predictions
    response_parts = []
    response_parts.append("**🎯 Intent Detection Results**\n\n")
    response_parts.append(f"**Query:** *{user_query}*\n")
    response_parts.append(f"**Processing Time:** {processing_time:.2f}s\n\n")
    response_parts.append("---\n\n")
    
    # All Predictions sorted by score (including unknown intents)
    response_parts.append("**📊 All Predictions (sorted by score):**\n\n")
    sorted_scores = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (intent, score) in enumerate(sorted_scores, 1):
        bar_length = int(score * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        marker = "✓" if intent in known_intents else " "
        response_parts.append(f"[{marker}] **{rank}. {intent}** — {bar} {score:.3f}\n")
    
    # Add unknown intents at the end (they don't have confidence scores)
    if unknown_intents:
        for intent in unknown_intents:
            rank = len(sorted_scores) + 1
            response_parts.append(f"[✓] **{rank}. {intent}** — 🆕 NEW (LLM extracted)\n")
            rank += 1
    
    response_parts.append("\n---\n\n")
    response_parts.append("**Is this correct?** Select an action below:\n")
    
    # Actions for user validation
    actions = [
        cl.Action(
            name="correct",
            payload={"action": "correct"},
            label="✅ Correct",
            description="Predictions are correct"
        ),
        cl.Action(
            name="incorrect",
            payload={"action": "incorrect"},
            label="❌ Incorrect - Let me fix",
            description="I want to modify the intents"
        )
        # ,
        # cl.Action(
        #     name="export_json",
        #     payload={
        #         "query": user_query,
        #         "confidence_scores": confidence_scores,
        #         "known_intents": known_intents,
        #         "checkpoint_id": checkpoint_id
        #     },
        #     label="📥 Export JSON",
        #     description="Download results as JSON"
        # )
    ]
    
    await cl.Message(content="".join(response_parts), actions=actions).send()


@cl.action_callback("correct")
async def on_correct(action: cl.Action):
    """User confirms predictions are correct"""
    pending = cl.user_session.get("pending_classification")
    if pending:
        # Save feedback to memory - user confirmed the predictions are correct
        try:
            memory = get_intent_memory()
            success = memory.add_feedback(
                checkpoint_id=pending['checkpoint_id'],
                correct_intents=pending['known_intents'],
                feedback_notes="User confirmed predictions as correct"
            )
            feedback_status = "✅ Saved to memory" if success else "⚠️ Could not save to memory"
        except Exception as e:
            feedback_status = f"⚠️ Error saving: {str(e)}"
        
        # Combine known intents and unknown intents for display
        all_final_intents = list(pending['known_intents']) + list(pending.get('unknown_intents', []))
        
        await cl.Message(
            content=f"✅ **Confirmed!** Intents saved.\n\n"
                    f"**Final Intents:** {', '.join(all_final_intents) if all_final_intents else 'None'}\n"
                    f"**Checkpoint ID:** `{pending['checkpoint_id']}`\n"
                    f"**Memory Status:** {feedback_status}"
        ).send()
        
        # Store in test results
        test_results = cl.user_session.get("test_results", [])
        test_results.append({
            "query": pending["query"],
            "final_intents": all_final_intents,
            "was_corrected": False,
            "checkpoint_id": pending["checkpoint_id"]
        })
        cl.user_session.set("test_results", test_results)
        cl.user_session.set("pending_classification", None)


async def show_selection_interface(pending: dict):
    """Helper function to display the intent selection interface"""
    confidence_scores = pending["confidence_scores"]
    selected_intents = pending.get("selected_intents", [])
    custom_unknown = pending.get("custom_unknown_intents", [])
    
    # Get ALL non-canonical intents from confidence_scores (not just original predictions)
    all_noncanonical = [i for i in confidence_scores.keys() if i not in CANONICAL_INTENTS]
    
    # Build intent selection interface
    response_parts = []
    response_parts.append("**🔧 Select the Correct Intents**\n\n")
    response_parts.append(f"**Query:** *{pending['query']}*\n\n")
    response_parts.append("---\n\n")
    
    # Show ALL canonical intents with scores (sorted by score)
    response_parts.append("**📋 Canonical Intents** :\n\n")
    
    canonical_sorted = sorted(
        [(intent, confidence_scores.get(intent, 0)) for intent in CANONICAL_INTENTS],
        key=lambda x: x[1], reverse=True
    )
    for intent, score in canonical_sorted:
        is_selected = intent in selected_intents
        marker = "✅" if is_selected else "⬜"
        response_parts.append(f"{marker} **{intent}** (score: {score:.3f})\n")
    
    # Show ALL non-canonical intents from confidence_scores (sorted by score)
    noncanonical_sorted = []
    if all_noncanonical:
        response_parts.append("\n**⚠️ Non-Canonical Intents** (learned by model):\n\n")
        noncanonical_sorted = sorted(
            [(intent, confidence_scores.get(intent, 0)) for intent in all_noncanonical],
            key=lambda x: x[1], reverse=True
        )
        for intent, score in noncanonical_sorted:
            is_selected = intent in selected_intents
            marker = "✅" if is_selected else "⬜"
            response_parts.append(f"{marker} 🔶 **{intent}** (score: {score:.3f})\n")
    
    # Show custom unknown intents added by user
    if custom_unknown:
        response_parts.append("\n**🆕 Custom Unknown Intents** (click to toggle):\n\n")
        for intent in custom_unknown:
            response_parts.append(f"🔷 **{intent}**\n")
    
    response_parts.append("\n---\n\n")
    
    # Show current selection with categories
    canonical_selected = [i for i in selected_intents if i in CANONICAL_INTENTS]
    noncanonical_selected = [j for j in selected_intents if j not in CANONICAL_INTENTS]
    
    response_parts.append("**Current Selection:**\n")
    if canonical_selected:
        response_parts.append(f"• Canonical: {', '.join(canonical_selected)}\n")
    if noncanonical_selected:
        response_parts.append(f"• Non-Canonical: {', '.join(noncanonical_selected)}\n")
    if custom_unknown:
        response_parts.append(f"• Custom Unknown: {', '.join(custom_unknown)}\n")
    if not canonical_selected and not noncanonical_selected and not custom_unknown:
        response_parts.append("None\n")
    
    # Create actions for ALL intents (canonical + non-canonical)
    actions = []
    
    # IMPORTANT: Add confirm and add_unknown buttons FIRST so they always show
    actions.append(cl.Action(
        name="confirm_selection",
        payload={"action": "confirm"},
        label="✅ Confirm Selection"
    ))
    
    actions.append(cl.Action(
        name="add_unknown",
        payload={"action": "add_unknown"},
        label="➕ Add Unknown Intent"
    ))
    
    # First add canonical intents (sorted by score)
    for intent, _ in canonical_sorted:
        is_selected = intent in selected_intents
        actions.append(cl.Action(
            name="toggle_intent",
            payload={"intent": intent, "type": "canonical"},
            label=f"{'✅' if is_selected else '⬜'} {intent}"
        ))
    
    # Then add ALL non-canonical intents from confidence_scores (sorted by score)
    for intent, _ in noncanonical_sorted:
        is_selected = intent in selected_intents
        actions.append(cl.Action(
            name="toggle_intent",
            payload={"intent": intent, "type": "noncanonical"},
            label=f"{'✅' if is_selected else '⬜'} 🔶 {intent}"
        ))
    
    # Add actions to toggle custom unknown intents
    for intent in custom_unknown:
        actions.append(cl.Action(
            name="toggle_unknown",
            payload={"intent": intent},
            label=f"🔷 {intent}"
        ))
    
    await cl.Message(content="".join(response_parts), actions=actions).send()


@cl.action_callback("incorrect")
async def on_incorrect(action: cl.Action):
    """User wants to modify intents - show selection interface"""
    pending = cl.user_session.get("pending_classification")
    if not pending:
        await cl.Message(content="⚠️ No pending classification to modify.").send()
        return
    
    await show_selection_interface(pending)


@cl.action_callback("toggle_intent")
async def on_toggle_intent(action: cl.Action):
    """Toggle an intent selection"""
    intent = action.payload.get("intent")
    if not intent:
        return
    
    pending = cl.user_session.get("pending_classification")
    if not pending:
        await cl.Message(content="⚠️ No pending classification.").send()
        return
    
    selected_intents = pending.get("selected_intents", [])
    
    # Toggle the intent
    if intent in selected_intents:
        selected_intents.remove(intent)
        status = "⬜ Removed"
    else:
        selected_intents.append(intent)
        status = "✅ Added"
    
    pending["selected_intents"] = selected_intents
    cl.user_session.set("pending_classification", pending)
    
        # Show categorized current selection
    custom_unknown = pending.get("custom_unknown_intents", [])
    canonical_selected = [i for i in selected_intents if i in CANONICAL_INTENTS]
    noncanonical_selected = [i for i in selected_intents if i not in CANONICAL_INTENTS]
    
    selection_parts = []
    if canonical_selected:
        selection_parts.append(f"Canonical: {', '.join(canonical_selected)}")
    if noncanonical_selected:
        selection_parts.append(f"Non-Canonical: {', '.join(noncanonical_selected)}")
    if custom_unknown:
        selection_parts.append(f"Custom Unknown: {', '.join(custom_unknown)}")
    
    selection_display = "\n• ".join(selection_parts) if selection_parts else "None"
    if selection_parts:
        selection_display = "• " + selection_display
    
    await cl.Message(
        content=f"{status}: **{intent}**\n\n"
                f"**Current selection:**\n{selection_display}"
    ).send()


@cl.action_callback("toggle_unknown")
async def on_toggle_unknown(action: cl.Action):
    """Toggle a custom unknown intent selection"""
    intent = action.payload.get("intent")
    if not intent:
        return
    
    pending = cl.user_session.get("pending_classification")
    if not pending:
        await cl.Message(content="⚠️ No pending classification.").send()
        return
    
    custom_unknown = pending.get("custom_unknown_intents", [])
    selected_intents = pending.get("selected_intents", [])
    
    # Toggle the unknown intent
    if intent in custom_unknown:
        custom_unknown.remove(intent)
        status = "⬜ Removed unknown"
    else:
        custom_unknown.append(intent)
        status = "✅ Added unknown"
    
    pending["custom_unknown_intents"] = custom_unknown
    cl.user_session.set("pending_classification", pending)
    
    # Show categorized current selection
    canonical_selected = [i for i in selected_intents if i in CANONICAL_INTENTS]
    noncanonical_selected = [i for i in selected_intents if i not in CANONICAL_INTENTS]
    
    selection_parts = []
    if canonical_selected:
        selection_parts.append(f"Canonical: {', '.join(canonical_selected)}")
    if noncanonical_selected:
        selection_parts.append(f"Non-Canonical: {', '.join(noncanonical_selected)}")
    if custom_unknown:
        selection_parts.append(f"Custom Unknown: {', '.join(custom_unknown)}")
    
    selection_display = "\n• ".join(selection_parts) if selection_parts else "None"
    if selection_parts:
        selection_display = "• " + selection_display
    
    await cl.Message(
        content=f"{status}: **{intent}**\n\n"
                f"**Current selection:**\n{selection_display}"
    ).send()


@cl.action_callback("remove_noncanonical")
async def on_remove_noncanonical(action: cl.Action):
    """Remove a non-canonical intent from selection"""
    intent = action.payload.get("intent")
    if not intent:
        return
    
    pending = cl.user_session.get("pending_classification")
    if not pending:
        await cl.Message(content="⚠️ No pending classification.").send()
        return
    
    selected_intents = pending.get("selected_intents", [])
    
    if intent in selected_intents:
        selected_intents.remove(intent)
        pending["selected_intents"] = selected_intents
        cl.user_session.set("pending_classification", pending)
        
        await cl.Message(
            content=f"❌ Removed non-canonical intent: **{intent}**\n\n"
                    f"*This was a learned intent from the model, not a canonical intent.*"
        ).send()


@cl.action_callback("remove_unknown")
async def on_remove_unknown(action: cl.Action):
    """Remove a custom unknown intent"""
    intent = action.payload.get("intent")
    if not intent:
        return
    
    pending = cl.user_session.get("pending_classification")
    if not pending:
        await cl.Message(content="⚠️ No pending classification.").send()
        return
    
    custom_unknown = pending.get("custom_unknown_intents", [])
    
    if intent in custom_unknown:
        custom_unknown.remove(intent)
        pending["custom_unknown_intents"] = custom_unknown
        cl.user_session.set("pending_classification", pending)
        
        await cl.Message(
            content=f"❌ Removed custom unknown intent: **{intent}**"
        ).send()


@cl.action_callback("add_unknown")
async def on_add_unknown(action: cl.Action):
    """User wants to add a custom unknown intent"""
    res = await cl.AskUserMessage(
        content="**Enter the unknown intent name(s):**\n\n"
                "Type intent name(s) separated by commas (e.g., `weather_impact, road_conditions`):",
        timeout=120
    ).send()
    
    if res:
        user_input = res['output'].strip()
        
        # Split by comma to handle multiple intents
        intent_parts = [part.strip() for part in user_input.split(',')]
        
        pending = cl.user_session.get("pending_classification")
        if pending:
            custom_unknown = pending.get("custom_unknown_intents", [])
            added_intents = []
            already_exists = []
            
            for part in intent_parts:
                if not part:
                    continue
                # Clean up: lowercase, replace spaces with underscores
                unknown_intent = part.lower().replace(' ', '_')
                
                if unknown_intent not in custom_unknown:
                    custom_unknown.append(unknown_intent)
                    added_intents.append(unknown_intent)
                else:
                    already_exists.append(unknown_intent)
            
            pending["custom_unknown_intents"] = custom_unknown
            cl.user_session.set("pending_classification", pending)
            
            # Build response message
            msg_parts = []
            if added_intents:
                msg_parts.append(f"✅ Added unknown intent(s): **{', '.join(added_intents)}**")
            if already_exists:
                msg_parts.append(f"⚠️ Already exists: {', '.join(already_exists)}")
            
            if msg_parts:
                await cl.Message(content="\n".join(msg_parts)).send()
            
            # Incrementally add new intents to centroids (background)
            if added_intents:
                try:
                    from utils.ML.ml_based_intent_classification.incremental_centroid_update import add_intent_to_centroids
                    import threading
                    
                    user_query = pending.get('query', '')
                    for new_intent in added_intents:
                        # Run in background thread to not block UI
                        thread = threading.Thread(
                            target=add_intent_to_centroids,
                            args=(new_intent, user_query),
                            daemon=True
                        )
                        thread.start()
                        print(f"🔄 Adding '{new_intent}' to centroids (background)")
                except Exception as e:
                    print(f"⚠️ Could not update centroids: {e}")
            
            # Refresh the selection interface with updated list
            if added_intents:
                await show_selection_interface(pending)


@cl.action_callback("confirm_selection")
async def on_confirm_selection(action: cl.Action):
    """User confirms their intent selection"""
    pending = cl.user_session.get("pending_classification")
    if not pending:
        await cl.Message(content="⚠️ No pending classification.").send()
        return
    
    selected_intents = pending.get("selected_intents", [])
    custom_unknown = pending.get("custom_unknown_intents", [])
    
    # Include ALL selected intents (canonical + non-canonical) + custom unknown
    # User explicitly selected these, so keep them all
    canonical_selected = [i for i in selected_intents if i in CANONICAL_INTENTS]
    noncanonical_selected = [i for i in selected_intents if i not in CANONICAL_INTENTS]
    final_intents = selected_intents + custom_unknown  # Keep ALL selected intents
    
    # Save feedback to memory
    try:
        memory = get_intent_memory()
        notes_parts = []
        if noncanonical_selected:
            notes_parts.append(f"Non-canonical: {noncanonical_selected}")
        if custom_unknown:
            notes_parts.append(f"Custom unknown: {custom_unknown}")
        feedback_notes = "User corrected intents. " + ". ".join(notes_parts) if notes_parts else "User corrected intents"
        success = memory.add_feedback(
            checkpoint_id=pending['checkpoint_id'],
            correct_intents=final_intents,
            feedback_notes=feedback_notes
        )
        feedback_status = "✅ Saved to memory" if success else "⚠️ Could not save to memory"
    except Exception as e:
        feedback_status = f"⚠️ Error saving: {str(e)}"
    
    # Build final result message
    response_parts = []
    response_parts.append("✅ **Selection Confirmed!**\n\n")
    response_parts.append(f"**Query:** *{pending['query']}*\n\n")
    response_parts.append("---\n\n")
    
    if canonical_selected:
        response_parts.append("**Selected Canonical Intents:**\n")
        for intent in canonical_selected:
            score = pending['confidence_scores'].get(intent, 0)
            response_parts.append(f"• {intent} (score: {score:.3f})\n")
    
    if noncanonical_selected:
        response_parts.append("\n**Selected Non-Canonical Intents:**\n")
        for intent in noncanonical_selected:
            score = pending['confidence_scores'].get(intent, 0)
            response_parts.append(f"• 🔶 {intent} (score: {score:.3f})\n")
    
    if not canonical_selected and not noncanonical_selected:
        response_parts.append("**No intents selected.**\n")
    
    if custom_unknown:
        response_parts.append("\n**Custom Unknown Intents:**\n")
        for intent in custom_unknown:
            response_parts.append(f"• 🔷 {intent}\n")
    
    response_parts.append(f"\n**Final Intent List:** {final_intents}")
    response_parts.append(f"\n**Checkpoint ID:** `{pending['checkpoint_id']}`")
    response_parts.append(f"\n**Memory Status:** {feedback_status}")
    
    await cl.Message(content="".join(response_parts)).send()
    
    # Store in test results
    test_results = cl.user_session.get("test_results", [])
    test_results.append({
        "query": pending["query"],
        "original_intents": pending.get("known_intents", []),
        "final_intents": final_intents,
        "custom_unknown_intents": custom_unknown,
        "was_corrected": True,
        "checkpoint_id": pending["checkpoint_id"]
    })
    cl.user_session.set("test_results", test_results)
    cl.user_session.set("pending_classification", None)


@cl.action_callback("export_json")
async def on_export_json(action: cl.Action):
    """Handle JSON export action"""
    data = action.payload
    await cl.Message(
        content=f"```json\n{json.dumps(data, indent=2)}\n```"
    ).send()


@cl.on_chat_end
async def end():
    """Session cleanup and summary"""
    query_count = cl.user_session.get("query_count", 0)
    test_results = cl.user_session.get("test_results", [])
    
    if query_count > 0:
        summary = "**📊 Session Summary**\n\n"
        summary += f"**Total Queries Tested:** {query_count}\n\n"
        
        if test_results:
            corrected_count = sum(1 for r in test_results if r.get("was_corrected", False))
            summary += f"**Corrections Made:** {corrected_count}/{len(test_results)}\n\n"
            
            summary += "**Results:**\n"
            for i, result in enumerate(test_results, 1):
                status = "📝 Corrected" if result.get("was_corrected") else "✅ Accepted"
                summary += f"\n{i}. *{result['query'][:50]}...* [{status}]\n"
                summary += f"   Intents: {', '.join(result.get('final_intents', []))}\n"
        
        await cl.Message(content=summary).send()
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
