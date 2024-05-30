# test
import os
import ollama
from transformers import Trainer, TrainingArguments

# Step 1: Load the pre-trained model
model_name = 'llama3'
model = ollama.models.load(model_name)

# Step 2: Load your specific dataset
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines

train_data = load_dataset('/path/to/your/train.txt')
val_data = load_dataset('/path/to/your/val.txt')

# Convert datasets to appropriate format
train_dataset = ollama.datasets.TextDataset(train_data)
val_dataset = ollama.datasets.TextDataset(val_data)

# Step 3: Set up the data collator
data_collator = ollama.data.DataCollatorForLanguageModeling(model)

# Step 4: Set up the training configuration
training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    overwrite_output_dir=True,       # Overwrite the content of the output directory
    num_train_epochs=3,              # Number of training epochs
    per_device_train_batch_size=4,   # Batch size for training
    per_device_eval_batch_size=4,    # Batch size for evaluation
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=10,
)

# Step 5: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Step 6: Fine-tune the model
trainer.train()

# Step 7: Save the fine-tuned model
model.save_pretrained('./fine-tuned-llama3')
print("Model fine-tuning completed and saved locally.")
