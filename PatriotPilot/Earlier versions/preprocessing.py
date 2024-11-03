import json
import re
import string
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources if you haven't already
nltk.download('punkt')
nltk.download('stopwords')

# Load stopwords from NLTK
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back into a single string
    preprocessed_text = " ".join(tokens)
    
    return preprocessed_text

# Function to preprocess the entire content from a JSON page
def preprocess_entire_page(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Join all text fields into a single block of text
    page_text = ""

    # Iterate through all key-value pairs in the JSON
    for key, value in data.items():
        if isinstance(value, str):  # If the value is a string, add it to the page_text
            page_text += value + " "
        elif isinstance(value, list):  # If the value is a list, add each item (assuming strings) to the page_text
            for item in value:
                if isinstance(item, str):
                    page_text += item + " "
                elif isinstance(item, dict):  # If item is a dictionary, recursively process it
                    for sub_key, sub_value in item.items():
                        if isinstance(sub_value, str):
                            page_text += sub_value + " "

    # Preprocess the entire block of text
    preprocessed_page_text = preprocess_text(page_text)
    
    return preprocessed_page_text

# Example usage
if __name__ == "__main__":
    # Preprocess the entire page content from the JSON file
    preprocessed_page_text = preprocess_entire_page('contact_information.json')

    # Save preprocessed text to a file for embedding
    with open('preprocessed_page_text.txt', 'w') as outfile:
        outfile.write(preprocessed_page_text)

    # Print a sample of the preprocessed text
    print(preprocessed_page_text[:500])  # Display the first 500 characters of the preprocessed text
