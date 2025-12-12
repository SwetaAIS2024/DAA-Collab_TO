"""
Standalone script to train the intent classification model.
Run this BEFORE deploying the application.

Usage:
    # Train with original data only
    python DAA-Collab/utils/ML/ml_based_intent_classification/standalone_training.py
    
    # Train with original data + validated memory interactions
    python DAA-Collab/utils/ML/ml_based_intent_classification/standalone_training.py --include-memory
"""

import argparse
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(os.path.join(BASE_DIR, "DAA-Collab"))

from utils.ML.ml_based_intent_classification.model_transformer import train_intent_classifier

def main():
    parser = argparse.ArgumentParser(description="Train Intent Classification Model")
    parser.add_argument(
        '--include-memory',
        action='store_true',
        help='Include validated interactions from LangGraph memory'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("INTENT CLASSIFICATION MODEL TRAINING")
    print("="*80)
    print(f"Include memory data: {args.include_memory}\n")
    
    try:
        train_intent_classifier(include_memory_data=args.include_memory)
        print("\n✅ Training completed successfully!")
        print("   Model is ready for production use.\n")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()