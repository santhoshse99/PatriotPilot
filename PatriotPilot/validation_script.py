import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import PeftModel, LoraConfig, TaskType
from datasets import load_dataset
from accelerate import Accelerator

# Initialize the Accelerator
accelerator = Accelerator()

# File paths
model_dir = "./Meta-Llama-3.1-8B"
adapter_dir = "./lora_llama_finetuned"
dataset_file = "./instruction_finetuning_data_final_combined.json"

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

# Load LoRA adapter
model = PeftModel.from_pretrained(model, adapter_dir).to(accelerator.device)

# Load the fine-tuning dataset with validation split
dataset = load_dataset('json', data_files=dataset_file, split='train[:90%]')
val_dataset = load_dataset('json', data_files=dataset_file, split='train[90%:]')

def preprocess_function(examples):
    inputs = [f"Instruction: {instr}\nContext: {ctx}\nResponse:" for instr, ctx in zip(examples['instruction'], examples['context'])]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding='longest')
    labels = tokenizer(examples['response'], max_length=256, truncation=True, padding='longest')['input_ids']

    model_inputs['labels'] = labels
    return model_inputs

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)

# Use Data Collator for Language Modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./lora_llama_finetuned',
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=10,
    learning_rate=5e-5,
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

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits, shift_labels)

        if return_outputs:
            return loss, outputs
        return loss

# Initialize the CustomTrainer with training and validation datasets
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_val_dataset,  # Validation dataset
    data_collator=data_collator,
)

# Start fine-tuning
print("Starting fine-tuning...")
trainer.train()

# Evaluate after training
print("Evaluating the model...")
metrics = trainer.evaluate(eval_dataset=tokenized_val_dataset)
print(f"Validation metrics: {metrics}")

# Save the fine-tuned model
model.save_pretrained('./lora_llama_finetuned')
tokenizer.save_pretrained('./lora_llama_finetuned')
