import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load the chunked data from preprocessed_chunks.json
with open('preprocessed_chunks.json', 'r') as file:
    chunks = json.load(file)

# Load the embedding model (E5-large-v2)
embedding_model = SentenceTransformer('intfloat/e5-large-v2')

# Embed the text chunks
embeddings = embedding_model.encode(chunks)

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
    json.dump(chunks, f)

print("Embeddings created and stored in FAISS successfully!")
