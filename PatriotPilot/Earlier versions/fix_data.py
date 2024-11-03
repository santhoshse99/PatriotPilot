import json

# File paths
input_file = 'instruction_finetuning_data_final_combined.json'
output_file = 'instruction_finetuning_data_fixed.json'

# Load the JSON data
with open(input_file, 'r') as f:
    data = json.load(f)

# Ensure all response fields are strings
for entry in data:
    if isinstance(entry['response'], list):
        # Join list elements into a single string
        entry['response'] = ' '.join(entry['response'])
    elif not isinstance(entry['response'], str):
        # Convert non-string types to strings
        entry['response'] = str(entry['response'])

# Save the fixed data
with open(output_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Fixed data saved to {output_file}.")
