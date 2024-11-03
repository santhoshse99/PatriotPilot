import json
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources if you haven't already
nltk.download('punkt')
nltk.download('stopwords')

# Load stopwords from NLTK
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    preprocessed_text = " ".join(tokens)
    return preprocessed_text

# Function to flatten and label JSON data
def flatten_and_label_json(data, parent_key=''):
    flat_text = ""
    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{parent_key} {key}".strip()
            flat_text += flatten_and_label_json(value, full_key)
    elif isinstance(data, list):
        for item in data:
            flat_text += flatten_and_label_json(item, parent_key)
    elif isinstance(data, str):
        labeled_text = f"{parent_key}: {data}"
        flat_text += labeled_text + " | "
    return flat_text

# Function to preprocess the entire content from a JSON page and split into chunks
def preprocess_entire_page(file_path, max_chunk_size=512):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Flatten the JSON structure and retain context markers
    flattened_text = flatten_and_label_json(data)

    # Preprocess the flattened text
    preprocessed_text = preprocess_text(flattened_text)

    # Split the preprocessed text into smaller chunks
    text_chunks = [
        preprocessed_text[i:i+max_chunk_size]
        for i in range(0, len(preprocessed_text), max_chunk_size)
    ]

    return text_chunks

# Main function to process multiple files
if __name__ == "__main__":
    files = [
        '../StructuredWebscrapeData/cs_advising.json',
        '../StructuredWebscrapeData/cs_contact_info.json',
        '../StructuredWebscrapeData/cs_people_directory.json'
    ]

    all_chunks = []
    for file_path in files:
        chunks = preprocess_entire_page(file_path)
        all_chunks.extend(chunks)

    # Save all chunks to a file for verification
    with open('preprocessed_chunks.json', 'w') as outfile:
        json.dump(all_chunks, outfile)

    # Print a sample of the preprocessed chunks for verification
    print(f"Sample of preprocessed chunks: {all_chunks[:5]}")
