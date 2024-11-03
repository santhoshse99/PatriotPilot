import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load the preprocessed text data
with open('preprocessed_page_text.txt', 'r') as file:
    preprocessed_text = file.read()

# Split the preprocessed text into smaller chunks for embedding (optional)
# This helps in cases where the text is very large. For simplicity, we split by sentence here.
text_chunks = preprocessed_text.split(". ")  # You can split based on sentence or other delimiters

# Load the embedding model (E5-large-v2)
embedding_model = SentenceTransformer('intfloat/e5-large-v2')

# Embed the text chunks
embeddings = embedding_model.encode(text_chunks)

# Convert embeddings to a NumPy array
embeddings = np.array(embeddings)

# Initialize FAISS index
embedding_dimension = embeddings.shape[1]  # The size of the embeddings (number of dimensions)
faiss_index = faiss.IndexFlatL2(embedding_dimension)  # L2 distance index for similarity search

# Add embeddings to the FAISS index
faiss_index.add(embeddings)

# Save the FAISS index to a file for future use
faiss.write_index(faiss_index, 'faiss_index.index')

# Save the metadata (original text chunks) to a JSON file
with open('metadata.json', 'w') as f:
    json.dump(text_chunks, f)

print("Embeddings created and stored in FAISS successfully!")
