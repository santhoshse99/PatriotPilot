import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from accelerate import Accelerator

# Initialize the Accelerator
accelerator = Accelerator()

# File paths
model_dir = "E:/University/Fall 2024/PatriotPilot/Meta-Llama-3.1-8B"
dataset_file = 'instruction_finetuning_data_fixed.json'

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model weights on CUDA
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    local_files_only=True,
    trust_remote_code=True,
)

# Prepare model with Accelerator
model = accelerator.prepare(model)

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
    inputs = [f"Instruction: {instr}\nContext: {ctx}\nResponse:" for instr, ctx in zip(examples['instruction'], examples['context'])]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding='max_length')
    labels = tokenizer(examples['response'], max_length=256, truncation=True, padding='max_length')['input_ids']

    # Ensure input and label lengths match
    model_inputs['labels'] = labels
    print(f"Model inputs batch size: {len(model_inputs['input_ids'])}, Labels batch size: {len(model_inputs['labels'])}")
    return model_inputs

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Use Data Collator for Language Modeling to handle padding and shifting
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./lora_llama_finetuned',
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=1e-4,
    logging_steps=10,
    evaluation_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    report_to='none',
)

# Custom Trainer to handle loss computation debugging
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Debug shapes of logits and labels
        print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")

        # Shift logits and labels for loss calculation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Ensure batch sizes match
        assert shift_logits.size(0) == shift_labels.size(0), "Mismatch in batch size"

        # Calculate loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss

# Initialize the CustomTrainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['train'],
    data_collator=data_collator,
)

# Start fine-tuning
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete!")

# Save the fine-tuned model
model.save_pretrained('./lora_llama_finetuned')
tokenizer.save_pretrained('./lora_llama_finetuned')
