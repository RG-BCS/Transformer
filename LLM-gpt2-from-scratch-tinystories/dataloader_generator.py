# Dataset for project: TinyStories from Hugging Face
# Contains multiple independent short stories.

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast

# Constants
SEED = 20
BATCH_SIZE = 32
MAX_LENGTH = 128
loss_ignore_index = -100  # index to ignore in loss calculation

# Set random seed for reproducibility
torch.manual_seed(SEED)

# Load TinyStories dataset (subset "10M_1", train split)
dataset = load_dataset("eminorhan/tinystories", "10M_1", split="train")

# Load GPT2 tokenizer and add padding token
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

vocab_size = len(tokenizer)       # Vocabulary size including added tokens
PAD_token = tokenizer.pad_token_id
EOS_token = tokenizer.eos_token_id


def tokenize_function(example, MAX_LENGTH):
    """
    Tokenize text with padding and truncation to fixed length.
    
    Args:
        example (dict): Example containing "text" key.
        MAX_LENGTH (int): Maximum token length.

    Returns:
        Tokenized output as PyTorch tensors.
    """
    return tokenizer(
        example["text"],
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )


class TinyStoryDataset(Dataset):
    """
    PyTorch Dataset wrapping TinyStories dataset for language modeling.
    """

    def __init__(self, dataset, tokenizer, max_length=MAX_LENGTH):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]

        # Tokenize text with +1 length to allow shifting labels for LM
        enc = self.tokenizer(
            text,
            max_length=self.max_length + 1,
            padding="max_length",
            truncation=True,
            return_tensors=None  # returns lists not tensors
        )
        input_ids = enc["input_ids"]

        # Prepare input and target sequences shifted by one token for LM training
        input_tensor = torch.tensor(input_ids[:-1])  # all tokens except last
        label_tensor = torch.tensor(input_ids[1:])   # all tokens except first (shifted)

        # Mask loss on padding tokens by setting labels to ignore_index (-100)
        label_tensor[input_tensor == self.pad_token_id] = loss_ignore_index

        return input_tensor, label_tensor


def collate_fn(batch):
    """
    Collate function to batch input tensors and labels.
    Computes lengths of sequences ignoring padding tokens.

    Args:
        batch (list): List of tuples (input_tensor, label_tensor)

    Returns:
        Tuple of batched input_ids, labels, and input_lengths
    """
    input_ids, labels = zip(*batch)  # unzip list of tuples into two lists
    input_ids = torch.stack(input_ids)  # stack into tensor (B, L)
    labels = torch.stack(labels)        # stack into tensor (B, L)

    pad_token_id = tokenizer.pad_token_id
    input_lengths = (input_ids != pad_token_id).sum(dim=1)  # lengths per sequence

    return input_ids, labels, input_lengths


# Instantiate Dataset and DataLoader
train_ds = TinyStoryDataset(dataset, tokenizer, max_length=MAX_LENGTH)
train_dl = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    shuffle=True
)
