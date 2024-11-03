import requests
from bs4 import BeautifulSoup
import json

url = 'https://cs.gmu.edu/about/contact-info/'
response = requests.get(url, verify= False)

soup = BeautifulSoup(response.text, 'html.parser')

data = {
    "contact_information": [],
    "leadership_information": [],
    "staff_information": []
}


contactInformation = soup.find_all('p')

print("Contact Information:")
for info in contactInformation:
    text = info.get_text().strip()
    if text:
        data["contact_information"].append(text)

# Scrape the leadership table (Leadership contact details)
print("\n Administrative Contacts:")
print("\nLeadership Information:")
leadership_table = soup.find('table', height="160")
if leadership_table:
    rows = leadership_table.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        if len(cols) == 2:  # Ensure each row has two columns (Name and Position)
            name = cols[0].get_text().strip()
            position = cols[1].get_text().strip()
            data["leadership_information"].append({"name": name, "position": position})

# Scrape the staff table (Staff contact details)
print("\nStaff Information:")
staff_table = soup.find('table', height="164")
if staff_table:
    rows = staff_table.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        if len(cols) == 2:  # Ensure each row has two columns (Name and Position)
            name = cols[0].get_text().strip()
            position = cols[1].get_text().strip()
            data["staff_information"].append({"name": name, "position": position})



with open('contact_information.json', 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)