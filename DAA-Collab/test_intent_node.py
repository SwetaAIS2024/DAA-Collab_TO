"""
Unit tests for Intent Classification Node

Usage:
    python test_intent_node.py "Your custom query"                # Test single query
    python test_intent_node.py "Query 1" "Query 2" "Query 3"      # Test multiple queries
"""

import sys
import os
from pathlib import Path

# Disable progress bars
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from func.state.agent_state import AgentState
from func.nodes.intent_classification_node import intent_classification


def test_query(query: str, test_name: str = "Custom Query"):
    """Test classification for a given query"""
    print(f"\nQuery: {query}")
    print("-" * 60)
    
    state = AgentState(instruction=query)
    result = intent_classification(state)
    
    print(f"\n{'='*20} RESULTS {'='*20}\n")
    print(f"Classified Intents ({len(result.all_intents_extracted)}):")
    for intent in result.all_intents_extracted:
        score = result.intent_confidence_scores.get(intent, 0.0)
        print(f"  - {intent}: {score:.4f}")
    
    print(f"\nKnown Intents ({len(result.known_intents)}): {result.known_intents}")
    print(f"Unknown Intents ({len(result.unknown_intents)}): {result.unknown_intents}")
    
    return result.classification_success


def run_queries(queries: list):
    """Run tests for user-provided queries"""
    
    for query in queries:
        test_query(query)
    
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # User provided queries
        queries = sys.argv[1:]
        success = run_queries(queries)
        sys.exit(0 if success else 1)
    else:
        print("\nUsage: python test_intent_node.py \"Query 1\" \"Query 2\" ...")
        print("Please provide at least one query as an argument.\n")
        sys.exit(1)
