import json
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
model_dir = "E:/University/Fall 2024/PatriotPilot/Meta-Llama-3.1-8B"
#bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the FAISS index and metadata
try:
    faiss_index = faiss.read_index('E:/University/Fall 2024/PatriotPilot/WebScraper/faiss_index.index')
except Exception as e:
    print(f"Error loading FAISS index: {e}")

try:
    with open('E:/University/Fall 2024/PatriotPilot/WebScraper/metadata.json', 'r') as f:
        metadata = json.load(f)
except Exception as e:
    print(f"Error loading metadata: {e}")

# Load the embedding model (E5-large-v2)
embedding_model = SentenceTransformer('intfloat/e5-large-v2')

llama_tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
llama_model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config, 
    local_files_only=True,
    trust_remote_code=True
    )
#).to("cuda")

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
    inputs = llama_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    # Generate a response using LLaMA
    outputs = llama_model.generate(**inputs, max_new_tokens=100, num_return_sequences=1)

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
