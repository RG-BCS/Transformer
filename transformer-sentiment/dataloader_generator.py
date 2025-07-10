import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter, OrderedDict

# Set random seed for reproducibility
SEED = 40
np.random.seed(SEED)
torch.manual_seed(SEED)

# Global constants
BATCH_SIZE = 32
MAX_SEQ_LEN = 256

# Tokenizer and vocabulary will be built after reading data
token_counts = Counter()
vocab = {}

def tokenizer(text):
    """
    Tokenizes input text by:
    - Removing HTML tags
    - Extracting emoticons
    - Removing punctuation
    - Lowercasing and splitting
    """
    text = re.sub('<[^>]*>', '', text)  # Remove HTML tags
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub(r'[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text.split()

def word_to_int(text, label, max_seq_len=MAX_SEQ_LEN):
    """
    Converts text and label to integer format using vocabulary mapping.
    """
    tokens = tokenizer(text)[:max_seq_len]
    token_ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    return token_ids, label_to_int(label)

def label_to_int(label):
    """
    Converts sentiment label to integer.
    """
    return 1.0 if label.lower() == 'positive' else 0.0

class MyDataset(Dataset):
    """
    Custom dataset that stores raw text and labels.
    """
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

def collate_batch(batch, max_seq_len=MAX_SEQ_LEN):
    """
    Pads and batches a list of samples for DataLoader.
    Returns:
        padded_inputs: Tensor of shape (batch_size, max_len)
        labels: Tensor of shape (batch_size,)
        lengths: Tensor of original sequence lengths
    """
    labels, inputs, lengths = [], [], []

    for text, label in batch:
        input_ids, label_id = word_to_int(text, label, max_seq_len)
        input_tensor = torch.tensor(input_ids, dtype=torch.int64)
        inputs.append(input_tensor)
        labels.append(label_id)
        lengths.append(len(input_ids))

    padded_inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    return padded_inputs, torch.tensor(labels), torch.tensor(lengths)

# === Dataset Preparation ===

# Load dataset
dataset = pd.read_csv("hf://datasets/scikit-learn/imdb/IMDB Dataset.csv")

# Create train/valid/test splits
indices = np.arange(len(dataset))
train_indices = np.random.choice(indices, 25000, replace=False)
test_indices = [idx for idx in indices if idx not in train_indices]

train_df = dataset.iloc[train_indices].reset_index(drop=True)
train_df, valid_df = train_df[:20000], train_df[20000:]

# Extract text and labels
train_texts = train_df['review'].values
train_labels = train_df['sentiment'].values
valid_texts = valid_df['review'].values
valid_labels = valid_df['sentiment'].values

# Build vocabulary from training data
for text in train_texts:
    tokens = tokenizer(text)
    token_counts.update(tokens)

# Sort by frequency and add special tokens
sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
sorted_tokens.insert(0, ("<pad>", 0))
sorted_tokens.insert(1, ("<unk>", 1))
ordered_vocab = OrderedDict(sorted_tokens)
vocab = {word: idx for idx, (word, _) in enumerate(ordered_vocab.items())}

# Create Dataset and DataLoader
train_dataset = MyDataset(train_texts, train_labels)
valid_dataset = MyDataset(valid_texts, valid_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
