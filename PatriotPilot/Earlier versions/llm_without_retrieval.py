import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from peft import PeftModel

# Model directory and device configuration
model_dir = "E:/University/Fall 2024/PatriotPilot/Meta-Llama-3.1-8B"
adapter_dir = "./lora_llama_finetuned"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def generate_response_without_context(query):
    # Prepare the prompt without any retrieved context
    prompt = f"Answer this question: {query}\nAnswer:"

    # Tokenize the input
    inputs = llama_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    # Generate a response using the fine-tuned model
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
        
        # Generate a response using LLaMA without any retrieved information
        llama_response = generate_response_without_context(user_query)
        
        # Display the generated response
        print("\nLLaMA's Response:")
        print(llama_response)
        print("\n--- End of Response ---\n")
