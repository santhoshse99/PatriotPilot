import json
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Load the FAISS index and metadata
faiss_index = faiss.read_index('faiss_index.index')

with open('metadata.json', 'r') as f:
    metadata = json.load(f)

# Load the embedding model (E5-large-v2)
embedding_model = SentenceTransformer('intfloat/e5-large-v2')

# Load the LLaMA tokenizer and model
llama_tokenizer = AutoTokenizer.from_pretrained("E:/University/Fall 2024/PatriotPilot/Meta-Llama-3.1-8B")
llama_model = AutoModelForCausalLM.from_pretrained("E:/University/Fall 2024/PatriotPilot/Meta-Llama-3.1-8B")

def search_faiss(query):
    # Embed the user's query
    query_embedding = embedding_model.encode([query])

    # Search the FAISS index for the closest matches (k=3 to get the top 3 results)
    D, I = faiss_index.search(np.array(query_embedding), k=3)

    # Retrieve the corresponding text chunks
    retrieved_texts = [metadata[i] for i in I[0]]
    
    return retrieved_texts

def generate_response(query, retrieved_texts):
    # Concatenate retrieved texts into a single context
    context = " ".join(retrieved_texts)

    # Prepare the prompt for LLaMA
    prompt = f"Based on the following information: {context}\n\nQuestion: {query}\nAnswer:"

    # Tokenize the input
    inputs = llama_tokenizer(prompt, return_tensors="pt")

    # Generate a response using LLaMA
    outputs = llama_model.generate(**inputs, max_length=100, num_return_sequences=1)

    # Decode and return the generated response
    response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Allow dynamic input during script execution
if __name__ == "__main__":
    while True:
        # Get user input
        user_query = input("Enter your query (or type 'exit' to quit): ").strip()
        
        if user_query.lower() == "exit":
            break
        
        # Retrieve relevant text chunks
        retrieved_texts = search_faiss(user_query)
        
        # Generate a response using LLaMA based on the retrieved information
        llama_response = generate_response(user_query, retrieved_texts)
        
        # Display the generated response
        print("\nLLaMA's Response:")
        print(llama_response)
        print("\n--- End of Response ---\n")
