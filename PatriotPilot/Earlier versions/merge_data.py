import json

# File paths for existing and new instruction files
previous_combined_file = 'instruction_finetuning_data_combined.json'
part2_file = 'instruction_finetuning_data_part2.json'
people_directory_file = 'instruction_finetuning_data_people_directory.json'

# Load the previous combined data
with open(previous_combined_file, 'r') as f:
    combined_data = json.load(f)

# Load the part2 instruction data
with open(part2_file, 'r') as f:
    part2_data = json.load(f)

# Load the people directory instruction data
with open(people_directory_file, 'r') as f:
    people_data = json.load(f)

# Merge all three lists
merged_data = combined_data + part2_data + people_data

# Save the final merged data as a new JSON file
final_output_file = 'instruction_finetuning_data_final_combined.json'
with open(final_output_file, 'w') as f:
    json.dump(merged_data, f, indent=2)

print(f"Final merged data saved to {final_output_file}. Total instructions: {len(merged_data)}.")
