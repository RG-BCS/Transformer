"""
dataloader_generator.py

This module handles dataset loading, text preprocessing, tokenization, vocabulary construction,
and DataLoader generation for the GoEmotions dataset. It supports emoji handling and multi-label outputs.

Author: [Your Name]
"""

from datasets import load_dataset
import emoji
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import re
from collections import Counter, OrderedDict

# Load GoEmotions dataset from HuggingFace
dataset = load_dataset("go_emotions")
label_names = dataset['train'].features['labels'].feature.names

# Constants and hyperparameters
SEED = 25
np.random.seed(SEED)
torch.manual_seed(SEED)
BATCH_SIZE = 32
MAX_SEQ_LEN = 256
PAD_token = 0
UNK_token = 1


class TextProcessor:
    """
    Processes raw text by tokenizing, cleaning, building vocabulary, and handling emoji.
    """
    def __init__(self, text_dataset, data_type='train', dataset_for_vocab=True, MAX_SEQ_LEN=None,
                 PAD_token=PAD_token, UNK_token=UNK_token):
        self.text_dataset = text_dataset
        self.data_type = data_type
        self.dataset_for_vocab = dataset_for_vocab
        self.MAX_SEQ_LEN = MAX_SEQ_LEN
        self.PAD_token = PAD_token
        self.UNK_token = UNK_token

        if self.dataset_for_vocab:
            self.vocab = self.build_vocabulary()

    def build_vocabulary(self):
        """
        Builds a vocabulary dictionary from training and validation sets.
        Returns:
            vocab: Dict mapping token to integer index
        """
        assert self.data_type in ['train', 'validation'], "Vocabulary can only be built using train or validation set"
        assert self.dataset_for_vocab, "Vocabulary building flag must be True"

        token_counts = Counter()
        for split in ['train', 'validation']:
            for text in self.text_dataset[split]['text']:
                tokens = self.tokenizer(text)
                token_counts.update(tokens)

        sorted_by_frequency = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_by_frequency.insert(0, ("<pad>", self.PAD_token))
        sorted_by_frequency.insert(1, ("<unk>", self.UNK_token))

        ord_dict = OrderedDict(sorted_by_frequency)
        vocab = dict(zip(list(ord_dict.keys()), range(len(ord_dict))))

        self.vocab_emotions = self.text_dataset[self.data_type].features['labels'].feature.names
        self.inv_vocab = {v: k for k, v in vocab.items()}  # Create inverse mapping

        return vocab

    def tokenizer(self, text):
        """
        Tokenizes input text with emoji demojization and basic cleaning.

        Args:
            text (str): Raw input text

        Returns:
            List of cleaned tokens
        """
        text = emoji.demojize(text, delimiters=(" ", " "))
        emoticon_pattern = r'(?::|;|=)(?:-)?(?:\)|\(|D|P|O)'
        text = re.sub(f'({emoticon_pattern})', r' \1 ', text)
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9:\s_()\-]', '', text)
        tokens = text.split()
        return tokens

    def word_to_int(self, text):
        """
        Converts text into list of token IDs.

        Args:
            text (str): Raw input string

        Returns:
            List of integer token IDs
        """
        tokenized = self.tokenizer(text)
        return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokenized]

    def int_to_word(self, indices):
        """
        Converts a list or tensor of token IDs back into text tokens.

        Args:
            indices (list[int] or Tensor): Sequence of indices

        Returns:
            List of token strings
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return [self.inv_vocab.get(idx, "<unk>") for idx in indices]

    def collate_batch(self, batch):
        """
        Collate function for DataLoader to handle variable-length sequences and multi-labels.

        Args:
            batch: List of tuples [(token_ids, label_indices), ...]

        Returns:
            Tuple: padded_text_list, label_tensor, lengths_tensor
        """
        text_list, label_list, lengths = [], [], []
        for text_int_form, label_ in batch:
            if self.MAX_SEQ_LEN is not None:
                text_int_form = text_int_form[:self.MAX_SEQ_LEN]
            text_tensor = torch.tensor(text_int_form, dtype=torch.int64)
            text_list.append(text_tensor)
            lengths.append(len(text_tensor))

            multi_hot = torch.zeros(28)
            multi_hot[label_] = 1.0  # multi-label binary format
            label_list.append(multi_hot)

        padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
        label_tensor = torch.stack(label_list)
        lengths_tensor = torch.tensor(lengths)

        return padded_text_list, label_tensor, lengths_tensor


class EmotionDataset(Dataset):
    """
    Custom dataset wrapper for GoEmotions, returns tokenized text and label indices.
    """
    def __init__(self, dataset, text_processor_, data_type):
        self.dataset = dataset
        self.data_type = data_type
        self.text_processor_ = text_processor_

    def __len__(self):
        return len(self.dataset[self.data_type])

    def __getitem__(self, index):
        """
        Fetch a single sample from the dataset.

        Returns:
            Tuple of (tokenized_text_ids, label_indices)
        """
        sample = self.dataset[self.data_type][index]
        text = sample['text']
        label_indices = sample['labels']
        token_ids = self.text_processor_.word_to_int(text)
        return token_ids, label_indices


# Initialize processing pipeline
text_processor = TextProcessor(dataset, data_type='train', dataset_for_vocab=True, MAX_SEQ_LEN=MAX_SEQ_LEN)

# Dataset objects
train_ds = EmotionDataset(dataset, text_processor, 'train')
valid_ds = EmotionDataset(dataset, text_processor, 'validation')
test_ds = EmotionDataset(dataset, text_processor, 'test')

# DataLoaders with padding & masking
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=text_processor.collate_batch)
valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=text_processor.collate_batch)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=text_processor.collate_batch)
