import torch
import torch.distributed as dist
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import os

# Set up environment variables for distributed training
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

# Initialize the process group
dist.init_process_group(backend='gloo', rank=0, world_size=1)

# File paths
model_dir = "E:/University/Fall 2024/PatriotPilot/Meta-Llama-3.1-8B"
dataset_file = 'instruction_finetuning_data_fixed.json'

# Check if CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure CUDA is properly installed and accessible.")

# Set device to CUDA
device = torch.device("cuda:0")
device_map = infer_auto_device_map(model, max_memory={0: "10GB", "cpu": "30GB"})

# Load the base model and tokenizer directly on CUDA
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model weights on CUDA
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    local_files_only=True,
    trust_remote_code=True,
).to(device)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Clear the CUDA cache
torch.cuda.empty_cache()

# Wrap the model with FSDP and enable mixed-precision
model = FSDP(model, process_group=dist.group.WORLD, mixed_precision=True)

# Configure LoRA for fine-tuning
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Load the fine-tuning dataset
dataset = load_dataset('json', data_files=dataset_file)

def preprocess_function(examples):
    instructions = examples['instruction']
    contexts = examples['context']
    responses = examples['response']
    
    inputs = [f"Instruction: {instr}\nContext: {ctx}\nResponse:" for instr, ctx in zip(instructions, contexts)]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding=True)  # Reduced max length
    labels = tokenizer(responses, max_length=128, truncation=True, padding=True)['input_ids']
    model_inputs['labels'] = labels

    return model_inputs

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./lora_llama_finetuned',
    per_device_train_batch_size=1,  # Reduced batch size
    per_device_eval_batch_size=2,  # Reduced batch size
    gradient_accumulation_steps=8,  # Adjusted for lower batch size
    num_train_epochs=3,
    learning_rate=1e-4,
    logging_steps=10,
    evaluation_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    report_to='none',
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['train'],
)

# Start fine-tuning
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete!")

# Save the fine-tuned model
model.save_pretrained('./lora_llama_finetuned')
tokenizer.save_pretrained('./lora_llama_finetuned')

# Clean up the process group
dist.destroy_process_group()
