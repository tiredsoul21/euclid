from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch

# Load the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token  # Setting pad token to eos token

# Prepare your dataset
inputs = [
    "What are some effective classroom management strategies?",
    "How can I engage students who are disinterested?",
    "What are the best practices for remote learning?"
]

outputs = [
    "Establish clear rules and routines to set expectations.",
    "Incorporate interactive activities to capture their attention.",
    "Use various tools and platforms to keep lessons engaging."
]

# Tokenization with attention masks
train_encodings = tokenizer(inputs, truncation=True, padding=True, return_tensors='pt')
labels = tokenizer(outputs, truncation=True, padding=True, return_tensors='pt')

# Convert to PyTorch Dataset
class RedditDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels['input_ids'][idx]  # Using input_ids from labels directly
        return item

    def __len__(self):
        return len(self.labels['input_ids'])

train_dataset = RedditDataset(train_encodings, labels)

# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=50,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    logging_steps=10,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Generating responses
input_text = "What are the best ways to teach math to elementary students?"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)

# Generate responses
output = model.generate(
    input_ids,
    max_length=50,
    num_return_sequences=5,
    do_sample=True,  # Enables sampling
    top_k=50,
    top_p=0.95
)

# Decode the outputs
responses = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(len(output))]
for response in responses:
    print(response)