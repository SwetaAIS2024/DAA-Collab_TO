import csv
import json
from collections import Counter
import ast
import os

# Read the CSV file
path_prompt_dataset = 'DAA-Collab/data/MCP_tool_code_generation_dataset/User_Input/code_01/dataset.csv'
with open(path_prompt_dataset, 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

# Open output file for writing
output_path = 'DAA-Collab/data/MCP_tool_code_generation_dataset/Code_Generator_Output/code_01/Code_Output.txt'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w') as out:
    out.write("=" * 80 + "\n")
    out.write("DATASET ANALYSIS: " + path_prompt_dataset + "\n")
    out.write("=" * 80 + "\n")

    # Basic stats
    out.write(f"\nTotal Prompts: {len(data)}\n")

    # Parse intents and collect statistics
    all_intents = []
    empty_intents_count = 0
    intent_per_prompt = []

    for row in data:
        try:
            # Parse the intent string (it's a list as string)
            intents = ast.literal_eval(row['intents'])
            if not intents:
                empty_intents_count += 1
            all_intents.extend(intents)
            intent_per_prompt.append(len(intents))
        except:
            empty_intents_count += 1
            intent_per_prompt.append(0)

    # Unique intents
    unique_intents = sorted(set(all_intents))
    out.write(f"\nUnique Intent Labels Found: {len(unique_intents)}\n")
    out.write("\nIntent Labels:\n")
    for i, intent in enumerate(unique_intents, 1):
        out.write(f"  {i}. {intent}\n")

    # Intent frequency
    out.write("\n" + "=" * 80 + "\n")
    out.write("INTENT FREQUENCY DISTRIBUTION\n")
    out.write("=" * 80 + "\n")
    intent_counts = Counter(all_intents)
    for intent, count in intent_counts.most_common():
        percentage = (count / len(data)) * 100
        out.write(f"  {intent:30s} : {count:4d} occurrences ({percentage:5.1f}% of prompts)\n")

    # Multi-intent analysis
    out.write("\n" + "=" * 80 + "\n")
    out.write("MULTI-INTENT ANALYSIS\n")
    out.write("=" * 80 + "\n")
    out.write(f"  Prompts with 0 intents: {empty_intents_count} ({empty_intents_count/len(data)*100:.1f}%)\n")
    out.write(f"  Prompts with 1 intent:  {intent_per_prompt.count(1)} ({intent_per_prompt.count(1)/len(data)*100:.1f}%)\n")
    out.write(f"  Prompts with 2 intents: {intent_per_prompt.count(2)} ({intent_per_prompt.count(2)/len(data)*100:.1f}%)\n")
    out.write(f"  Prompts with 3 intents: {intent_per_prompt.count(3)} ({intent_per_prompt.count(3)/len(data)*100:.1f}%)\n")
    out.write(f"  Prompts with 4 intents: {intent_per_prompt.count(4)} ({intent_per_prompt.count(4)/len(data)*100:.1f}%)\n")
    out.write(f"  Prompts with 5+ intents: {sum(1 for x in intent_per_prompt if x >= 5)} ({sum(1 for x in intent_per_prompt if x >= 5)/len(data)*100:.1f}%)\n")
    out.write(f"\n  Average intents per prompt: {sum(intent_per_prompt)/len(data):.2f}\n")
    out.write(f"  Max intents in single prompt: {max(intent_per_prompt)}\n")

    # Show examples with empty intents
    out.write("\n" + "=" * 80 + "\n")
    out.write("SAMPLE PROMPTS WITH EMPTY INTENTS\n")
    out.write("=" * 80 + "\n")
    empty_count = 0
    for row in data:
        try:
            intents = ast.literal_eval(row['intents'])
            if not intents:
                empty_count += 1
                if empty_count <= 10:
                    out.write(f"  ID {row['id']:3s}: {row['prompt'][:70]}...\n")
        except:
            pass

    # Show examples of multi-intent prompts
    out.write("\n" + "=" * 80 + "\n")
    out.write("SAMPLE PROMPTS WITH 4+ INTENTS\n")
    out.write("=" * 80 + "\n")
    multi_count = 0
    for row in data:
        try:
            intents = ast.literal_eval(row['intents'])
            if len(intents) >= 4:
                multi_count += 1
                if multi_count <= 10:
                    out.write(f"  ID {row['id']:3s}: {intents}\n")
                    out.write(f"        {row['prompt'][:70]}...\n")
                    out.write("\n")
        except:
            pass

    # Intent co-occurrence analysis
    out.write("\n" + "=" * 80 + "\n")
    out.write("TOP INTENT COMBINATIONS (2+ intents)\n")
    out.write("=" * 80 + "\n")
    combinations = []
    for row in data:
        try:
            intents = ast.literal_eval(row['intents'])
            if len(intents) >= 2:
                combinations.append(tuple(sorted(intents)))
        except:
            pass

    combo_counts = Counter(combinations)
    for combo, count in combo_counts.most_common(15):
        if count > 1:
            out.write(f"  {count:3d}x: {', '.join(combo)}\n")

    out.write("\n" + "=" * 80 + "\n")

print(f"Analysis complete! Output written to: {output_path}")