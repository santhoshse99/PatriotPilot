import json

# Load the people directory JSON file
people_file = '../StructuredWebscrapeData/cs_people_directory.json'
faq_people_file = '../FAQ/faq_people_directory.json'

# Create a list to hold instruction-response pairs
instruction_data = []

def create_instruction(entry, question, answer_key):
    """
    Helper function to create instruction-response pairs.
    """
    context = ', '.join([f"{k}: {v}" for k, v in entry.items()])
    response = entry.get(answer_key, "Information not available")
    return {
        "instruction": question,
        "context": context,
        "response": response
    }

# Load and process the people directory data
with open(people_file, 'r') as f:
    people_data = json.load(f)

# Iterate through each entry in the people directory
if 'faculty' in people_data:
    for faculty in people_data['faculty']:
        if 'name' in faculty and 'email' in faculty:
            instruction_data.append(
                create_instruction(faculty, f"What is {faculty['name']}'s email address?", 'email')
            )
        if 'name' in faculty and 'phone' in faculty:
            instruction_data.append(
                create_instruction(faculty, f"What is {faculty['name']}'s phone number?", 'phone')
            )
        if 'name' in faculty and 'office' in faculty:
            instruction_data.append(
                create_instruction(faculty, f"Where is {faculty['name']}'s office located?", 'office')
            )
        if 'name' in faculty and 'title' in faculty:
            instruction_data.append(
                create_instruction(faculty, f"What is {faculty['name']}'s position?", 'title')
            )
        if 'name' in faculty and 'webpage' in faculty:
            instruction_data.append(
                create_instruction(faculty, f"What is the webpage for {faculty['name']}?", 'webpage')
            )

# Load and process the FAQ people directory data
with open(faq_people_file, 'r') as f:
    faq_people_data = json.load(f)

# Add FAQ entries to the instruction data
for faq in faq_people_data:
    if 'question' in faq and 'answer' in faq:
        instruction_data.append({
            "instruction": faq['question'],
            "context": "",
            "response": faq['answer']
        })

# Save the new instruction-response data to a JSON file
output_file = 'instruction_finetuning_data_people_directory.json'
with open(output_file, 'w') as f:
    json.dump(instruction_data, f, indent=2)

print(f"Data processing for people directory complete. Saved to {output_file}.")
