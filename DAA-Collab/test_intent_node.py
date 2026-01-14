"""
Unit tests for Intent Classification Node

Usage:
    python test_intent_node.py "Your custom query"                # Test single query
    python test_intent_node.py "Query 1" "Query 2" "Query 3"      # Test multiple queries
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from func.state.agent_state import AgentState
from func.nodes.intent_classification_node import intent_classification


def test_query(query: str, test_name: str = "Custom Query"):
    """Test classification for a given query"""
    print("\n" + "="*60)
    print(f"TEST: {test_name}")
    print("="*60)
    print(f"Query: {query}")
    print("-" * 60)
    
    state = AgentState(instruction=query)
    result = intent_classification(state)
    
    print(f"\n{'='*20} RESULTS {'='*20}")
    print(f"Success: {result.classification_success}")
    print(f"Execution ID: {result.execution_id}")
    print(f"\nClassified Intents ({len(result.predicted_classified_intents)}):")
    for intent in result.predicted_classified_intents:
        score = result.intent_confidence_scores.get(intent, 0.0)
        print(f"  - {intent}: {score:.4f}")
    
    print(f"\nKnown Intents ({len(result.known_intents)}): {result.known_intents}")
    print(f"Unknown Intents ({len(result.unknown_intents)}): {result.unknown_intents}")
    print(f"Needs Tool Generation: {result.needs_tool_generation}")
    
    if result.error_log:
        print(f"\nErrors: {result.error_log}")
    
    return result.classification_success


def run_queries(queries: list):
    """Run tests for user-provided queries"""
    print("\n" + "="*60)
    print("INTENT CLASSIFICATION NODE - QUERY TESTS")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for i, query in enumerate(queries, 1):
        try:
            success = test_query(query, test_name=f"Query {i}")
            if success:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ ERROR: {type(e).__name__}: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Queries: {len(queries)}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")
    print("="*60 + "\n")
    
    return failed == 0


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
