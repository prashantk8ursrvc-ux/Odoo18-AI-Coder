import json
import statistics
import os
from collections import Counter

file_path = 'odoo18_expert_sft_dataset.jsonl'

instructions = []
inputs = []
outputs = []

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        instructions.append(data.get('instruction', ''))
        inputs.append(data.get('input', ''))
        outputs.append(data.get('output', ''))

print(f"Total samples: {len(instructions)}")

# Analyze lengths (in characters)
inst_lens = [len(i) for i in instructions]
input_lens = [len(i) for i in inputs]
output_lens = [len(o) for o in outputs]

print(f"Instruction Length - Min: {min(inst_lens)}, Max: {max(inst_lens)}, Mean: {statistics.mean(inst_lens):.2f}")
print(f"Input Length - Min: {min(input_lens)}, Max: {max(input_lens)}, Mean: {statistics.mean(input_lens):.2f}")
print(f"Output Length - Min: {min(output_lens)}, Max: {max(output_lens)}, Mean: {statistics.mean(output_lens):.2f}")

# Check for duplicates
unique_instructions = set(instructions)
print(f"Unique instructions: {len(unique_instructions)}")
unique_outputs = set(outputs)
print(f"Unique outputs: {len(unique_outputs)}")

# Most common instructions
inst_counter = Counter(instructions)
print("\nMost common instructions:")
for inst, count in inst_counter.most_common(5):
    print(f"- {inst[:100]}... (Count: {count})")
