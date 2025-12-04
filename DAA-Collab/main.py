"""Minimal LangGraph-based CSV analysis agent.

Flow: User supplies --instruction and --file (CSV).
1. Planner node builds JSON plan of tool steps.
2. Execute node runs each step sequentially and prints raw concatenated output.

"""

from __future__ import annotations
import argparse
from func.runner.run import run

def main():
    parser = argparse.ArgumentParser(description="Minimal CSV analysis agent")
    parser.add_argument("--instruction", required=True, help="User instruction/query")
    parser.add_argument("--file", required=True, help="Path to CSV dataset")
    args = parser.parse_args()
    run(args.instruction, args.file)

if __name__ == "__main__":
    main()
