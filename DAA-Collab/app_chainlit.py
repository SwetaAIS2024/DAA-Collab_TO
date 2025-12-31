"""
Chainlit Web Interface for Intent Classification Node Testing
Allows team members to test intent classification through a chat interface
"""

import chainlit as cl
from func.state.agent_state import AgentState
from func.nodes.intent_classification_node import intent_classification
import time
import json
import asyncio
import gc

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@cl.on_stop
async def on_stop():
    """Called when server stops (e.g., during auto-reload with -w flag)"""
    # Aggressively clear GPU memory to prevent OOM during auto-reload
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()


@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    # Clear GPU cache at startup to prevent OOM during auto-reload
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    await cl.Message(
        content="👋 **Welcome to Intent Classification Testing Tool!**\n\n"
                "I'll help you test the intent classification system.\n\n"
                "**How to use:**\n"
                "- Type any traffic-related query\n"
                "- I'll analyze it and show detected intents\n"
                "- You can provide feedback using 👍/👎 buttons\n\n"
                "**Example queries:**\n"
                "- *Show me accidents on Highway 1 last week*\n"
                "- *Which road has most traffic violations?*\n"
                "- *Visualize incident trends for December*\n\n"
                "Try it now! 👇"
    ).send()
    
    # Store session stats
    cl.user_session.set("query_count", 0)
    cl.user_session.set("test_results", [])


@cl.on_message
async def main(message: cl.Message):
    """Process user query and classify intent"""
    
    # Get query
    user_query = message.content
    
    # Update session stats
    query_count = cl.user_session.get("query_count", 0) + 1
    cl.user_session.set("query_count", query_count)
    
    # Create processing message
    processing_msg = cl.Message(content="")
    await processing_msg.send()
    
    # Step 1: Initialize
    async with cl.Step(name="🔧 Initializing", type="tool") as step:
        start_time = time.time()
        state = AgentState(instruction=user_query)
        step.output = f"Created AgentState for query #{query_count}"
        await cl.sleep(0.5)  # Brief pause for UX
    
    # Step 2: Load Models
    async with cl.Step(name="📦 Loading Models", type="tool") as step:
        step.output = (
            "Loading embedding and LLM models...\n"
            "(First query may take 10-15 seconds)\n"
            "(Subsequent queries: ~3 seconds)"
        )
    
    # Step 3: Classify Intent
    async with cl.Step(name="🎯 Classifying Intent", type="llm") as step:
        try:
            # Run classification (blocking call wrapped in async)
            result = await asyncio.to_thread(intent_classification, state)
            
            # Check if classification succeeded
            if not result.classification_success:
                error_msg = "\n".join(result.error_log) if result.error_log else "Unknown error"
                step.output = f"❌ Classification failed: {error_msg}"
                await processing_msg.remove()
                await cl.Message(
                    content=f"**Classification Failed:**\n\n{error_msg}\n\n"
                            "Please check:\n"
                            "- Models are properly initialized\n"
                            "- Intent centroids are built\n"
                            "- Memory system is accessible"
                ).send()
                return
            
            # Extract results
            intent_dict = result.intent_classification
            confidence_scores = result.intent_confidence_scores
            known_intents = result.known_intents
            unknown_intents = result.unknown_intents
            checkpoint_id = result.mem_checkpoint_id
            
            # Validate extracted data
            if intent_dict is None or confidence_scores is None:
                step.output = "❌ Missing classification results"
                await processing_msg.remove()
                await cl.Message(
                    content="**Error:** Classification returned incomplete results.\n\n"
                            "Debug info:\n"
                            f"- intent_dict: {intent_dict}\n"
                            f"- confidence_scores: {confidence_scores}\n"
                            f"- known_intents: {known_intents}"
                ).send()
                return
            
            step.output = "✅ Classification completed successfully"
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            step.output = f"❌ Error during classification: {str(e)}"
            await processing_msg.remove()
            await cl.Message(
                content=f"**Error:** {str(e)}\n\n"
                        f"```\n{error_trace}\n```\n\n"
                        "Please check that models are properly initialized."
            ).send()
            return
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Remove processing message
    await processing_msg.remove()
    
    # Build response message
    response_parts = []
    
    # Header - use simple bold instead of ## heading
    response_parts.append("**🎯 Intent Classification Results**\n\n")
    response_parts.append(f"**Query:** *{user_query}*\n")
    response_parts.append(f"**Processing Time:** {processing_time:.2f}s\n")
    response_parts.append(f"**Checkpoint ID:** `{checkpoint_id}`\n")
    response_parts.append("\n---\n\n")
    
    # Known Intents
    if known_intents:
        response_parts.append("**✅ Known Intents**\n\n")
        for intent in known_intents:
            score = confidence_scores.get(intent, 0)
            response_parts.append(f"• **{intent}** (confidence: {score:.3f})\n")
    else:
        response_parts.append("**❌ No Known Intents Detected**\n\n")
    
    response_parts.append("\n")
    
    # Unknown Intents
    if unknown_intents:
        response_parts.append("**⚠️ Unknown Intents**\n\n")
        response_parts.append("*These intents were extracted by LLM and may require new capabilities:*\n\n")
        for intent in unknown_intents:
            response_parts.append(f"• **{intent}**\n")
    else:
        response_parts.append("**✅ No Unknown Intents**\n\n")
        response_parts.append("*All aspects covered by known intent classes*\n")
    
    response_parts.append("\n---\n\n")
    
    # Top 5 Confidence Scores
    response_parts.append("**📊 Top 5 Confidence Scores**\n\n")
    sorted_scores = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    
    for intent, score in sorted_scores:
        # Create progress bar
        bar_length = int(score * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        # Mark if selected
        marker = "✓" if intent in known_intents else " "
        
        response_parts.append(f"[{marker}] `{intent:25s}` {bar} {score:.3f}\n")
    
    # Combine response
    response_content = "".join(response_parts)
    
    # Send response with actions
    actions = [
        cl.Action(
            name="export_json",
            payload={
                "query": user_query,
                "intents": intent_dict,
                "confidence_scores": confidence_scores,
                "checkpoint_id": checkpoint_id,
                "processing_time": processing_time
            },
            label="📥 Export JSON",
            description="Download results as JSON"
        ),
        cl.Action(
            name="view_all_scores",
            payload=confidence_scores,
            label="📊 View All Scores",
            description="See all confidence scores"
        )
    ]
    
    # Send response with actions
    msg = await cl.Message(
        content=response_content,
        actions=actions
    ).send()
    
    # Store result for session
    test_results = cl.user_session.get("test_results", [])
    test_results.append({
        "query": user_query,
        "known_intents": known_intents,
        "unknown_intents": unknown_intents,
        "checkpoint_id": checkpoint_id
    })
    cl.user_session.set("test_results", test_results)


@cl.action_callback("export_json")
async def on_export_json(action: cl.Action):
    """Handle JSON export action"""
    data = action.payload  # Already a dict, no need to parse
    
    # Create downloadable file
    await cl.Message(
        content=f"```json\n{json.dumps(data, indent=2)}\n```\n\n"
                "Copy the JSON above or download it."
    ).send()


@cl.action_callback("view_all_scores")
async def on_view_all_scores(action: cl.Action):
    """Handle view all scores action"""
    scores = action.payload  # Already a dict, no need to parse
    
    # Sort by score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Build table
    content_parts = ["### 📊 All Confidence Scores\n\n"]
    content_parts.append("| Rank | Intent | Score | Bar |\n")
    content_parts.append("|------|--------|-------|-----|\n")
    
    for rank, (intent, score) in enumerate(sorted_scores, 1):
        bar_length = int(score * 10)
        bar = "█" * bar_length + "░" * (10 - bar_length)
        content_parts.append(f"| {rank} | {intent} | {score:.4f} | {bar} |\n")
    
    await cl.Message(content="".join(content_parts)).send()


@cl.on_chat_end
async def end():
    """Session cleanup and summary"""
    query_count = cl.user_session.get("query_count", 0)
    test_results = cl.user_session.get("test_results", [])
    
    if query_count > 0:
        summary = f"## 📊 Session Summary\n\n"
        summary += f"**Total Queries Tested:** {query_count}\n\n"
        
        if test_results:
            summary += "### Test Results:\n"
            for i, result in enumerate(test_results, 1):
                summary += f"\n**{i}.** *{result['query'][:50]}...*\n"
                summary += f"   - Known: {', '.join(result['known_intents']) if result['known_intents'] else 'None'}\n"
                summary += f"   - Unknown: {', '.join(result['unknown_intents']) if result['unknown_intents'] else 'None'}\n"
        
        summary += "\n\nThank you for testing! 👋"
        
        await cl.Message(content=summary).send()
    
    # Clear GPU cache when session ends
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
