# Intent Extraction Node - Integration README

## Overview

This package contains a production-ready Intent Classification Node for LangGraph agent systems. It classifies user queries into 10 canonical traffic-related intents and detects unknown intents using LLM.

## Quick Start

### 1. Install dependencies

```bash
pip install sentence-transformers torch transformers accelerate numpy scikit-learn joblib langgraph
```

### 2. Import and add to your graph

```python
from func.state.agent_state import AgentState
from func.nodes.intent_classification_node import intent_classification
from langgraph.graph import StateGraph

# Create graph
graph = StateGraph(AgentState)

# Add intent classification node
graph.add_node("intent_classification", intent_classification)

# Connect to your workflow
graph.set_entry_point("intent_classification")
graph.add_edge("intent_classification", "your_next_node")

# Compile and run
app = graph.compile()
result = app.invoke({"instruction": "Show me accidents on Highway 1"})
```

### 3. Access results

```python
result.predicted_classified_intents  # List of all intents
result.known_intents                 # Canonical intents found
result.unknown_intents               # New intents detected by LLM
result.intent_confidence_scores      # Dict of {intent: score}
result.classification_success        # True/False
result.needs_tool_generation         # True if unknown intents found
```

## Complete Integration Example

```python
from langgraph.graph import StateGraph, END
from func.state.agent_state import AgentState
from func.nodes.intent_classification_node import intent_classification

# Define your graph
def build_traffic_agent():
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("intent_classification", intent_classification)
    graph.add_node("data_retrieval", your_data_node)
    graph.add_node("analysis", your_analysis_node)
    graph.add_node("response", your_response_node)
    
    # Define flow
    graph.set_entry_point("intent_classification")
    
    # Route based on classification results
    graph.add_conditional_edges(
        "intent_classification",
        route_after_intent_classification,
        {
            "retrieve_data": "data_retrieval",
            "unknown_intent": "handle_unknown",
            "error": END
        }
    )
    
    graph.add_edge("data_retrieval", "analysis")
    graph.add_edge("analysis", "response")
    graph.add_edge("response", END)
    
    return graph.compile()

# Router function
def route_after_intent_classification(state: AgentState) -> str:
    if not state.classification_success:
        return "error"
    
    if state.needs_tool_generation:
        return "unknown_intent"
    
    return "retrieve_data"

# Run the agent
agent = build_traffic_agent()
result = agent.invoke({
    "instruction": "Show me traffic accidents on I-95 last week"
})

print(f"Intents: {result['predicted_classified_intents']}")
print(f"Confidence: {result['intent_confidence_scores']}")
```

## Canonical Intents (10)

1. **visualization** - Creating charts, graphs, plots
2. **incident_detection** - Finding/identifying incidents
3. **spatio_temporal** - Location and time-based queries
4. **meta_attributes** - Data properties (severity, type, etc.)
5. **traffic_impact** - Traffic flow, congestion analysis
6. **incident_classification** - Categorizing incident types
7. **traffic_anomaly** - Detecting unusual patterns
8. **causal_analysis** - Root cause analysis
9. **traffic_forecasting** - Predicting future traffic
10. **report_generation** - Creating reports/summaries

## State Fields

### INPUT (required):
- `state.instruction: str` - User query

### OUTPUT (populated by node):
- `state.intent_classification: Dict[str, List[str]]`
- `state.predicted_classified_intents: List[str]`
- `state.known_intents: List[str]`
- `state.unknown_intents: List[str]`
- `state.intent_confidence_scores: Dict[str, float]`
- `state.classification_success: bool`
- `state.needs_tool_generation: bool`
- `state.mem_checkpoint_id: str`
- `state.execution_id: str` (generated if not present)
- `state.error_log: List[str]` (on errors)

## Handling Unknown Intents

When unknown intents are detected (`state.unknown_intents` is not empty):

### 1. Check the flag:

```python
if state.needs_tool_generation:
    # Handle unknown intents
    for intent in state.unknown_intents:
        print(f"New intent detected: {intent}")
```

### 2. Options for handling:
- Generate new tools/capabilities
- Request user clarification
- Map to existing capabilities
- Add to training data for retraining

### 3. User feedback loop:

```python
from utils.memory.episodic import get_intent_memory

memory = get_intent_memory()
memory.add_feedback(
    checkpoint_id=state.mem_checkpoint_id,
    correct_intents=['incident_detection', 'new_intent'],
    feedback_notes="User confirmed intents"
)
```

## Routing Examples

### Example 1: Simple routing by intent

```python
def route_by_intent(state: AgentState) -> str:
    intents = state.predicted_classified_intents
    
    if 'visualization' in intents:
        return "create_visualization"
    elif 'incident_detection' in intents:
        return "detect_incidents"
    elif 'traffic_forecasting' in intents:
        return "forecast_traffic"
    else:
        return "general_analysis"

graph.add_conditional_edges(
    "intent_classification",
    route_by_intent,
    {
        "create_visualization": "viz_node",
        "detect_incidents": "incident_node",
        "forecast_traffic": "forecast_node",
        "general_analysis": "analysis_node"
    }
)
```

### Example 2: Multi-intent handling

```python
def handle_multiple_intents(state: AgentState) -> List[str]:
    """Process multiple intents in sequence"""
    nodes_to_execute = []
    
    for intent in state.predicted_classified_intents:
        if intent == 'incident_detection':
            nodes_to_execute.append('detect_incidents')
        if intent == 'visualization':
            nodes_to_execute.append('create_viz')
        if intent == 'report_generation':
            nodes_to_execute.append('generate_report')
    
    return nodes_to_execute or ['default_handler']
```

### Example 3: Confidence-based routing

```python
def route_by_confidence(state: AgentState) -> str:
    scores = state.intent_confidence_scores
    top_intent = max(scores.items(), key=lambda x: x[1])
    intent_name, confidence = top_intent
    
    if confidence < 0.6:
        return "request_clarification"
    elif confidence < 0.8:
        return "confirm_with_user"
    else:
        return f"execute_{intent_name}"
```

## Memory & Checkpoints

Each classification creates a checkpoint in memory:

### 1. Access checkpoint:

```python
checkpoint_id = state.mem_checkpoint_id
```

### 2. Retrieve history:

```python
from utils.memory.episodic import get_intent_memory

memory = get_intent_memory()
history = memory.get_all_interactions(limit=10)
```

### 3. Export for retraining:

```python
memory.export_validated_data()
# Creates CSV in data/memory/episodic/exports/
```

## Customization

### 1. Adjust thresholds in `intent_classification_node.py`:

```python
intent_threshold = 0.75  # Similarity threshold
min_intents = 3          # Minimum intents to return
```

### 2. Rebuild centroids with custom data:

```bash
python utils/ML/ml_based_intent_classification/build_centroids_script.py
```

### 3. Add new canonical intents:
- Update `data/user_prompt_dataset/traffic_prompts_realistic_option2_FIXED.csv`
- Add training examples for new intent
- Rebuild centroids

## Error Handling

The node handles errors gracefully:

```python
if not state.classification_success:
    print(f"Classification failed: {state.error_log}")
    # Handle error - retry, use fallback, notify user
```

### Common errors:
- **FileNotFoundError**: Centroid files missing (run `build_centroids_script.py`)
- **Model loading errors**: Check GPU/CUDA setup
- **Memory errors**: Reduce batch size or use CPU

## Performance

- **First run**: ~10-15 seconds (downloads models)
- **Subsequent runs**: ~2-3 seconds per query
- **GPU recommended** for production
- **Models cached in**: `utils/ML/ml_based_intent_classification/saved_models_transformer/`

## Models Used

### 1. Qwen/Qwen3-Embedding-8B (~7GB)
- **Purpose**: Generate query embeddings
- **Auto-downloaded** on first run

### 2. Qwen/Qwen2.5-7B-Instruct (~7GB)
- **Purpose**: Extract unknown intents
- **Auto-downloaded** when needed

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "FileNotFoundError: Model files not found" | Run `build_centroids_script.py` to generate centroids |
| "CUDA out of memory" | Set device to CPU or use smaller batch size |
| Unknown intents always empty | Check threshold settings, may be too permissive |
| Classification too slow | Use GPU, reduce debug printing, cache models |

## Testing Your Integration

### 1. Simple test:

```python
from func.state.agent_state import AgentState
from func.nodes.intent_classification_node import intent_classification

state = AgentState(instruction="Show me accidents on Highway 1")
result = intent_classification(state)

print(f"Success: {result.classification_success}")
print(f"Intents: {result.predicted_classified_intents}")
print(f"Scores: {result.intent_confidence_scores}")
```

### 2. In your graph:

```python
app = graph.compile()
test_queries = [
    "Show me traffic accidents",
    "Create a visualization of incidents",
    "Forecast traffic for tomorrow"
]

for query in test_queries:
    result = app.invoke({"instruction": query})
    print(f"Query: {query}")
    print(f"Intents: {result['predicted_classified_intents']}\n")
```

## Support & Questions

For issues or questions about integration:
1. Check this README
2. Review the example code above
3. Contact: [Add your contact information]

## Files Included

```
func/state/agent_state.py                    - State definition
func/nodes/intent_classification_node.py     - Main node
utils/ML/ml_based_intent_classification/
  intent_classification.py                   - Core logic
  build_centroids_script.py                  - Rebuild centroids
  saved_models_transformer/
    intent_centroids.joblib                  - Pre-computed centroids
    intent_classes.joblib                    - Intent classes
utils/common.py                              - Text preprocessing
utils/memory/episodic.py                     - Memory management
data/user_prompt_dataset/
  traffic_prompts_realistic_option2_FIXED.csv - Training data

NEXT STEPS
----------
1. Install dependencies
2. Copy this directory into your project
3. Import and add node to your graph
4. Run a test query
5. Implement routing logic based on intents
6. Add error handling
7. Customize thresholds if needed

Good luck with your integration!
