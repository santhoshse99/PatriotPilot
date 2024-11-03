import json
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from peft import PeftModel

# Model directory and device configuration
model_dir = "E:/University/Fall 2024/PatriotPilot/Meta-Llama-3.1-8B"
adapter_dir = "./lora_llama_finetuned"
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

# Load the LLaMA tokenizer and base model
llama_tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
llama_model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,  # Use FP16 for more efficient memory usage on the A100
    local_files_only=True,
    trust_remote_code=True
).to(device)

# Load the LoRA adapter and apply it to the base LLaMA model
llama_model = PeftModel.from_pretrained(llama_model, adapter_dir).to(device)

def search_faiss(query):
    # Embed the user's query
    query_embedding = embedding_model.encode([query])

    # Search the FAISS index for the closest matches (k=3 to get the top 3 results)
    D, I = faiss_index.search(np.array(query_embedding), k=3)

    # Retrieve the corresponding text chunks
    retrieved_texts = [metadata[i] for i in I[0]]
    
    return retrieved_texts

def generate_response(query, retrieved_texts, max_context_length=512):
    # Limit the number of characters for context to avoid overly long prompts
    context = " ".join(retrieved_texts)[:max_context_length]

    # Prepare the prompt for LLaMA
    prompt = f"Here is some information extracted from the dataset: {context}\n\nNow, answer this question based on the information extracted from the dataset:\n{query}\nAnswer:"

    # Tokenize the input
    inputs = llama_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    # Generate a response using the fine-tuned LLaMA with LoRA
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
