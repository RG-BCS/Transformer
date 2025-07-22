import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from transformers import get_linear_schedule_with_warmup

from model import OpenWebTextModel
from utils import generate_text_sample, train_gpt_2
from dataloader_generator import train_dl, tokenizer, vocab_size, PAD_token, EOS_token, loss_ignore_index, device

torch.manual_seed(1)

# Hyperparameters
learning_rate = 1e-4
num_epochs = 20
num_layers = 12
embed_size = 768     # adjust based on GPU memory
d_out_n_heads = embed_size
ffn_hidden_dim = 4 * embed_size
context_length = 256
num_heads = 12       # ensure embed_size % num_heads == 0

# Instantiate model
model = OpenWebTextModel(
    num_layers=num_layers,
    vocab_size=vocab_size,
    embed_size=embed_size,
    d_out_n_heads=d_out_n_heads,
    num_heads=num_heads,
    ffn_hidden_dim=ffn_hidden_dim,
    dropout=0.1,
    context_length=context_length,
    qkv_bias=False,
    PAD_token=PAD_token
).to(device)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

steps_per_epoch = len(train_dl)
total_steps = steps_per_epoch * num_epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss(ignore_index=loss_ignore_index)

# Train the model
total_loss = train_gpt_2(model, train_dl, num_epochs, loss_fn, optimizer, clip_norm=False, max_norm=1.0)

# Sample text prompts
sample_text = ["It was a dark and stormy night, and the wind howled through the trees",
               "The machine began to hum as the lights flickered on","Once upon a time"]

print(f"\nSample sentence:\n")

for text in sample_text:
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    methods = [('beam_search', {'beam_width': 3}), ('greedy', {}), ('multinomial', {}), 
        ('temperature_sampling', {'temperature': 0.8}),('top_k_sampling', {'top_k': 50, 'temperature': 0.7})]

    for m, params in methods:
        out_ids = generate_text_sample(model, input_ids, max_new_tokens=MAX_LENGTH,methods=m, eos_id=EOS_token, **params)
        gen_text = tokenizer.decode(out_ids[0].tolist())
        print(f"=== {m} ===\n{gen_text}\n")
    print("#" * 100)
    print("Next sample prompt begins")
    print("#" * 100)

plt.plot(total_loss)
plt.xlabel('Epoch');plt.ylabel('Loss');plt.title('Training Loss');plt.grid()
plt.show()

