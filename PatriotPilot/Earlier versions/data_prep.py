import json

# Paths for structured JSON and FAQ files
structured_file = '../StructuredWebscrapeData/cs_advising.json'
faq_file = '../FAQ/faq_contact_info.json'

# Create a list to hold instruction-response pairs
instruction_data = []

def create_instruction(entry, question, answer_key):
    """
    Helper function to create instruction-response pair.
    """
    context = ', '.join([f"{k}: {v}" for k, v in entry.items() if isinstance(v, str)])
    response = entry.get(answer_key, "Information not available")
    return {
        "instruction": question,
        "context": context,
        "response": response
    }

# Step 1: Process the structured JSON file
with open(structured_file, 'r') as f:
    data = json.load(f)

# Extracting contact information from structured JSON
contact_info = data.get('contact_info', {})

# Administration Graduate Advising location
if 'administration_graduate_advising' in contact_info:
    admin_grad_advising = contact_info['administration_graduate_advising']
    if 'location' in admin_grad_advising:
        instruction_data.append(
            create_instruction(admin_grad_advising, "Where is the Administration Graduate Advising office located?", 'location')
        )
    if 'map_url' in admin_grad_advising:
        instruction_data.append(
            create_instruction(admin_grad_advising, "What is the map URL for Administration Graduate Advising?", 'map_url')
        )

# Undergraduate Advising location
if 'undergraduate_advising' in contact_info:
    undergrad_advising = contact_info['undergraduate_advising']
    if 'location' in undergrad_advising:
        instruction_data.append(
            create_instruction(undergrad_advising, "Where is the Undergraduate Advising office located?", 'location')
        )
    if 'map_url' in undergrad_advising:
        instruction_data.append(
            create_instruction(undergrad_advising, "What is the map URL for Undergraduate Advising?", 'map_url')
        )

# Phone number
if 'phone_number' in contact_info:
    instruction_data.append(
        create_instruction(contact_info, "What is the department's phone number?", 'phone_number')
    )

# Mailing address
if 'mailing_address' in contact_info:
    mailing_address = contact_info['mailing_address']
    if 'department' in mailing_address:
        instruction_data.append(
            create_instruction(mailing_address, "What is the department's name in the mailing address?", 'department')
        )
    if 'university' in mailing_address:
        instruction_data.append(
            create_instruction(mailing_address, "Which university is mentioned in the mailing address?", 'university')
        )
    if 'address' in mailing_address:
        instruction_data.append(
            create_instruction(mailing_address, "What is the department's full mailing address?", 'address')
        )

# Email addresses
if 'email' in contact_info:
    email_info = contact_info['email']
    for program, email in email_info.items():
        instruction_data.append(
            {
                "instruction": f"What is the email address for the {program.replace('_', ' ')} program?",
                "context": f"{program.replace('_', ' ')} email: {email}",
                "response": email
            }
        )

# Leadership information
leadership_info = data.get('leadership_info', [])
for leader in leadership_info:
    if 'name' in leader and 'position' in leader:
        instruction_data.append(
            create_instruction(leader, f"What is {leader['name']}'s position?", 'position')
        )

# Staff information
staff_info = data.get('staff_info', [])
for staff in staff_info:
    if 'name' in staff and 'position' in staff:
        instruction_data.append(
            create_instruction(staff, f"What is {staff['name']}'s position?", 'position')
        )

# Step 2: Process the FAQ file
with open(faq_file, 'r') as f:
    faq_data = json.load(f)

# Extract instructions from the FAQ data
for faq in faq_data:
    instruction = faq.get('question', 'No question available')
    context = f"Source: {faq.get('source', 'Unknown')}, Category: {faq.get('category', 'General')}"
    response = faq.get('answer', 'No answer available')
    instruction_data.append({
        "instruction": instruction,
        "context": context,
        "response": response
    })

# Save the combined instruction data as a JSON file
output_file = 'instruction_finetuning_data_part1.json'
with open(output_file, 'w') as f:
    json.dump(instruction_data, f, indent=2)

print(f"Data preparation complete for cs_contact_info.json and its FAQ. Total instructions: {len(instruction_data)}. Saved to {output_file}.")
