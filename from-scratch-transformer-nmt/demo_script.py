# demo_script.py

import torch
import random
import matplotlib.pyplot as plt
import torch.nn as nn

from dataloader_generator import (
    prepareData, load_and_preprocess_data, TranslationDataset,
    collate_batch, PAD_token, SOS_token, EOS_token, MAX_LENGTH
)
from torch.utils.data import DataLoader
from model import Transformer
from utils import train_transformer
from transformers import get_linear_schedule_with_warmup

# Set manual seed for reproducibility
torch.manual_seed(13)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 32
learning_rate = 1e-3
num_epochs = 2000
num_layers = 2
embed_size = 256  # or 512
d_out_n_heads = embed_size
ffn_hidden_dim = 4 * embed_size  # transformer standard: FFN hidden dim = 4 * embed_size
num_heads = 4  # d_out_n_heads must be divisible by num_heads

# Load and prepare data
text_pairs = load_and_preprocess_data()

# Remove duplicate English sentences
seen_eng = set()
unique_text_pairs = []
for eng, spa in text_pairs:
    if eng not in seen_eng:
        unique_text_pairs.append((eng, spa))
        seen_eng.add(eng)
text_pairs = unique_text_pairs

input_lang, output_lang, pairs = prepareData('eng', 'spa', text_pairs)
eng_spa_ds = TranslationDataset(pairs, input_lang, output_lang)
train_dl = DataLoader(eng_spa_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

# Vocab sizes
src_vocab_size, target_vocab_size = input_lang.n_words, output_lang.n_words

# Initialize model
transformer_model = Transformer(
    num_layers=num_layers,
    src_vocab_size=src_vocab_size,
    target_vocab_size=target_vocab_size,
    embed_size=embed_size,
    d_out_n_heads=d_out_n_heads,
    num_heads=num_heads,
    ffn_hidden_dim=ffn_hidden_dim,
    dropout=0.5,
    context_length=MAX_LENGTH,
    qkv_bias=False,
    PAD_token=PAD_token
).to(device)

# Loss, optimizer, scheduler
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_token)
optimizer = torch.optim.Adam(transformer_model.parameters(), lr=learning_rate)
total_steps = len(train_dl) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# Train the model
train_loss = train_transformer(
    transformer_model,
    train_dl,
    num_epochs,
    loss_fn,
    optimizer,
    scheduler,
    device,
    input_lang,
    output_lang,
    clip_norm=True,
    max_norm=1.0,
    MAX_LENGTH=MAX_LENGTH,
    SOS_token=SOS_token,
    EOS_token=EOS_token
)

# Inference
print("\nSample Examples from the dataset and show results\n")
for _ in range(10):
    eng, spa = random.choice(text_pairs)
    print("Input:", eng)
    print("Target:", spa)
    result = transformer_model.generate(
        eng,
        input_lang,
        output_lang,
        max_len=MAX_LENGTH,
        SOS_token=SOS_token,
        EOS_token=EOS_token
    )
    predicted_tokens = result['output']
    predicted_sentence = " ".join([output_lang.index2word.get(idx, 'UNK') for idx in predicted_tokens])
    print("Predicted:", predicted_sentence)
    print("#" * 80)

# Plot training loss
plt.plot(train_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.title("Transformer Model Training Loss vs. Epoch")
plt.show()
