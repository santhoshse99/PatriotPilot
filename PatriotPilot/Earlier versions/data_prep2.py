import json

# File path for the cs_advising JSON file
file_path = '../StructuredWebscrapeData/cs_advising.json'

# List to hold instruction-response pairs
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

# Open and load the JSON file
with open(file_path, 'r') as f:
    data = json.load(f)

# Extracting undergraduate advising information
undergrad_advising = data.get('advising_information', {}).get('undergraduate_advising', {})

# Advisors' information
advisors = undergrad_advising.get('advisors', [])
for advisor in advisors:
    if 'name' in advisor and 'position' in advisor:
        instruction_data.append(
            create_instruction(advisor, f"What is {advisor['name']}'s position?", 'position')
        )
    if 'webpage' in advisor:
        instruction_data.append(
            create_instruction(advisor, f"What is {advisor['name']}'s webpage?", 'webpage')
        )

# Undergraduate Advising contact information
if 'contact' in undergrad_advising:
    contact_info = undergrad_advising['contact']
    if 'email' in contact_info:
        instruction_data.append(
            create_instruction(contact_info, "What is the undergraduate advising contact email?", 'email')
        )
    if 'phone' in contact_info:
        instruction_data.append(
            create_instruction(contact_info, "What is the undergraduate advising phone number?", 'phone')
        )

# Undergraduate Advising appointments
if 'appointment' in undergrad_advising:
    appointment_info = undergrad_advising['appointment']
    if 'note' in appointment_info:
        instruction_data.append(
            create_instruction(appointment_info, "What is the note for undergraduate advising appointments?", 'note')
        )
    if 'booking_link' in appointment_info:
        instruction_data.append(
            create_instruction(appointment_info, "Where can I book an undergraduate advising appointment?", 'booking_link')
        )

# Required Advising Appointments for Specific Courses
required_appointments = undergrad_advising.get('required_advising_appointments', [])
for appointment in required_appointments:
    if 'course' in appointment:
        instruction_data.append(
            create_instruction(appointment, f"What are the instructions for required advising for {appointment['course']}?", 'instructions')
        )
    if 'form_link' in appointment:
        instruction_data.append(
            create_instruction(appointment, f"Where can I find the advising form for {appointment['course']}?", 'form_link')
        )

# Extracting graduate advising information
grad_advising = data.get('advising_information', {}).get('graduate_advising', {})

# Graduate Advisors' information
grad_advisors = grad_advising.get('advisors', [])
for advisor in grad_advisors:
    if 'name' in advisor and 'position' in advisor:
        instruction_data.append(
            create_instruction(advisor, f"What is {advisor['name']}'s position?", 'position')
        )
    if 'webpage' in advisor:
        instruction_data.append(
            create_instruction(advisor, f"What is {advisor['name']}'s webpage?", 'webpage')
        )

# Graduate Advising contact information
if 'contact_information' in grad_advising:
    grad_contact_info = grad_advising['contact_information']
    if 'office_location' in grad_contact_info:
        instruction_data.append(
            create_instruction(grad_contact_info, "Where is the graduate advising office located?", 'office_location')
        )
    if 'email' in grad_contact_info:
        instruction_data.append(
            create_instruction(grad_contact_info, "What is the graduate advising contact email?", 'email')
        )
    if 'phone' in grad_contact_info:
        instruction_data.append(
            create_instruction(grad_contact_info, "What is the graduate advising phone number?", 'phone')
        )

# Extracting PhD advising information
phd_advising = data.get('advising_information', {}).get('phd_advising', {})

# PhD Advisor information
phd_advisor = phd_advising.get('advisor', {})
if 'name' in phd_advisor and 'position' in phd_advisor:
    instruction_data.append(
        create_instruction(phd_advisor, f"What is {phd_advisor['name']}'s position?", 'position')
    )
if 'webpage' in phd_advisor:
    instruction_data.append(
        create_instruction(phd_advisor, f"What is {phd_advisor['name']}'s webpage?", 'webpage')
    )

# PhD Advising contact information
if 'contact_information' in phd_advising:
    phd_contact_info = phd_advising['contact_information']
    if 'office_location' in phd_contact_info:
        instruction_data.append(
            create_instruction(phd_contact_info, "Where is the PhD advising office located?", 'office_location')
        )
    if 'email' in phd_contact_info:
        instruction_data.append(
            create_instruction(phd_contact_info, "What is the PhD advising contact email?", 'email')
        )
    if 'phone' in phd_contact_info:
        instruction_data.append(
            create_instruction(phd_contact_info, "What is the PhD advising phone number?", 'phone')
        )

# Save the prepared data as a JSON file for fine-tuning
output_file = 'instruction_finetuning_data_part2.json'
with open(output_file, 'w') as f:
    json.dump(instruction_data, f, indent=2)

print(f"Data preparation complete for cs_advising.json. Total instructions: {len(instruction_data)}. Saved to {output_file}.")
