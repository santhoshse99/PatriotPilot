import json
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

# Model directory and device configuration
model_dir = "E:/University/Fall 2024/PatriotPilot/Meta-Llama-3.1-8B"
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the FAISS index and metadata
try:
    faiss_index = faiss.read_index('faiss_index.index')
except Exception as e:
    print(f"Error loading FAISS index: {e}")

try:
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
except Exception as e:
    print(f"Error loading metadata: {e}")

# Load the embedding model (E5-large-v2)
embedding_model = SentenceTransformer('intfloat/e5-large-v2')

# Load the LLaMA tokenizer and model
llama_tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
llama_model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config, 
    local_files_only=True,
    trust_remote_code=True
)
#).to(device)

def search_all_instances(query, embeddings, metadata, similarity_threshold=0.7):
    # Embed the user's query
    query_embedding = embedding_model.encode([query])

    # Calculate cosine similarity between query embedding and all stored embeddings
    similarities = np.dot(embeddings, query_embedding.T) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    # Filter embeddings based on the similarity threshold
    relevant_indices = np.where(similarities >= similarity_threshold)[0]
    relevant_chunks = [metadata[i] for i in relevant_indices]

    return relevant_chunks

def format_for_llm(relevant_chunks):
    # Format the relevant chunks into structured text for LLaMA
    formatted_text = "Here is the gathered information:\n"
    for chunk in relevant_chunks:
        formatted_text += f"- {chunk}\n"
    return formatted_text

def generate_response(query, relevant_chunks, max_context_length=512):
    # Format the context for the LLM
    context = format_for_llm(relevant_chunks)[:max_context_length]

    # Prepare the prompt for LLaMA
    prompt = (
        f"{context}\n\nNow, answer the following question based on the information above:\n{query}\nAnswer:"
    )

    # Tokenize the input
    inputs = llama_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    # Generate a response using LLaMA
    outputs = llama_model.generate(
        **inputs,
        max_new_tokens=100,
        num_return_sequences=1,
        pad_token_id=llama_tokenizer.eos_token_id
    )

    # Decode and return the generated response
    response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Allow dynamic input during script execution
if __name__ == "__main__":
    # Load the embeddings from the FAISS index
    num_embeddings = faiss_index.ntotal
    embeddings = np.zeros((num_embeddings, faiss_index.d), dtype='float32')
    faiss_index.reconstruct_n(0, num_embeddings, embeddings)

    while True:
        # Get user input
        user_query = input("Enter your query (or type 'exit' to quit): ").strip()
        
        if user_query.lower() == "exit":
            break
        
        # Retrieve all relevant instances from the FAISS index
        relevant_chunks = search_all_instances(user_query, embeddings, metadata)
        
        # Generate a response using LLaMA based on the gathered information
        llama_response = generate_response(user_query, relevant_chunks)
        
        # Display the generated response
        print("\nLLaMA's Response:")
        print(llama_response)
        print("\n--- End of Response ---\n")
