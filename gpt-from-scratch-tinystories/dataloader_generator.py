import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)

# Constants
BATCH_SIZE = 32
MAX_LENGTH = 128
loss_ignore_index = -100

# Load TinyStories dataset from Hugging Face
dataset = load_dataset("eminorhan/tinystories", "10M_1", split="train")

# Load GPT-2 tokenizer and add a padding token if not present
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Vocabulary size for model embedding layers
vocab_size = len(tokenizer)

PAD_token = tokenizer.pad_token_id
EOS_token = tokenizer.eos_token_id

def tokenize_function(example, max_length=MAX_LENGTH):
    """
    Tokenize a single example text with truncation and padding.
    Returns a dictionary of tokenized outputs.
    """
    return tokenizer(
        example["text"],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

class TinyStoryDataset(Dataset):
    """
    Dataset wrapping the Hugging Face TinyStories dataset for language modeling.
    Prepares inputs and labels by shifting tokens for next-token prediction.
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

        # Tokenize text with padding and truncation, +1 for shifting labels
        enc = self.tokenizer(
            text,
            max_length=self.max_length + 1,
            padding="max_length",
            truncation=True,
            return_tensors=None
        )
        input_ids = enc["input_ids"]

        # Shift inputs and labels for causal LM (predict next token)
        input_tensor = torch.tensor(input_ids[:-1])
        label_tensor = torch.tensor(input_ids[1:])

        # Ignore loss on padding tokens
        label_tensor[input_tensor == self.pad_token_id] = LOSS_IGNORE_INDEX

        return input_tensor, label_tensor

def collate_fn(batch):
    """
    Collate function to batch sequences and compute input lengths.
    Pads sequences to MAX_LENGTH and computes non-pad token lengths.
    """
    input_ids, labels = zip(*batch)  # unzip list of tuples
    input_ids = torch.stack(input_ids)  # shape: (batch_size, seq_len)
    labels = torch.stack(labels)        # shape: (batch_size, seq_len)

    # Compute input lengths (number of non-pad tokens per sequence)
    input_lengths = (input_ids != PAD_token).sum(dim=1)

    return input_ids, labels, input_lengths

# Instantiate dataset and dataloader
train_ds = TinyStoryDataset(dataset, tokenizer, max_length=MAX_LENGTH)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

if __name__ == "__main__":
    # Quick sanity check: fetch one batch
    for batch in train_dl:
        input_ids, labels, input_lengths = batch
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Input lengths: {input_lengths}")
        break
