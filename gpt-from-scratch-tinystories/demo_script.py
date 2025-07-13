import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataloader_generator import train_dl, tokenizer, PAD_token, EOS_token, vocab_size, MAX_LENGTH, loss_ignore_index, device
from utils import train_tiny_gpt, generate_text_sample
from model import TinyStoryModel  # Assuming your model class is here

# Seed for reproducibility
SEED = 42
torch.manual_seed(SEED)

# Hyperparameters
learning_rate = 1e-4
num_epochs = 10
num_layers = 2
embed_size = 128     # Adjust depending on GPU capability
d_out_n_heads = embed_size
ffn_hidden_dim = 4 * embed_size
context_length = 256
num_heads = 4         # 4 or 8 depending on embed_size

# Instantiate the model
tiny_model = TinyStoryModel(num_layers, vocab_size, embed_size, d_out_n_heads, num_heads,
                            ffn_hidden_dim, dropout=0.1, context_length=context_length,
                            qkv_bias=False, PAD_token=PAD_token).to(device)

optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss(ignore_index=loss_ignore_index)

print("Starting training...")
total_loss = train_tiny_gpt(tiny_model, train_dl, num_epochs, loss_fn, optimizer,
                            clip_norm=False, max_norm=1.0, tokenizer=tokenizer, EOS_token=EOS_token)

# Sample generation from trained model
text = "Once upon a time"
print(f"\nSample sentence:-> {text} <-: is given to the trained model\n")

input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

methods = [
    ('greedy', {}),
    ('multinomial', {}),
    ('temperature_sampling', {'temperature': 0.8}),
    ('top_k_sampling', {'top_k': 50, 'temperature': 0.7})
]

for m, params in methods:
    out_ids = generate_text_sample(tiny_model, input_ids, max_new_tokens=MAX_LENGTH,
                                   methods=m, eos_id=EOS_token, **params)

    gen_text = tokenizer.decode(out_ids[0].tolist())
    print(f"=== {m} ===\n{gen_text}\n")

# Plot training loss
plt.plot(total_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
