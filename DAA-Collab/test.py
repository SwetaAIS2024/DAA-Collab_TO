from __future__ import annotations

from utils.ML.intent_extraction_classification.ml_intent_classification import train_intent_classifier
from func.state.agent_state import AgentState
from func.nodes.intent_classification_node import intent_classification
import argparse

def main():
    # Step 1: Train the ML model (one-time setup)

    train_intent_classifier()
    parser = argparse.ArgumentParser(description="Minimal CSV analysis agent")
    parser.add_argument("--instruction", required=True, help="User instruction/query")
    parser.add_argument("--file", required=True, help="Path to CSV dataset")
    args = parser.parse_args()

    # Test
    state = AgentState(args.instruction, args.file)

    result = intent_classification(state)

    print("Extracted keywords:", result.intent_extracted)
    print("Classified intents:", result.intent_classification)
    



if __name__ == "__main__":
    main()