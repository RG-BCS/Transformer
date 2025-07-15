import os
import re
import unicodedata
import torch
from torch.utils.data import Dataset, DataLoader

# Special tokens
PAD_token, SOS_token, EOS_token, UNK_token, NUM_SPECIAL_TOKENS = 0, 1, 2, 3, 4
MAX_LENGTH = 20  # Max sentence length
BATCH_SIZE = 64
DATA_PATH = "spa-eng/spa.txt"
ZIP_URL = "http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"

# Fix random seed
torch.manual_seed(42)

# Download and unzip the dataset
def maybe_download_data():
    if not os.path.exists("spa-eng"):
        os.system(f"wget -q {ZIP_URL}")
        os.system("unzip -q spa-eng.zip")

class LanguageDictionary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            PAD_token: 'PAD_token',
            SOS_token: 'SOS',
            EOS_token: 'EOS',
            UNK_token: 'UNK_token'
        }
        self.n_words = NUM_SPECIAL_TOKENS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?¿])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?¿]+", r" ", s)
    return s.strip()

def load_and_preprocess_data():
    maybe_download_data()
    with open(DATA_PATH, encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    pairs = []
    for line in lines:
        if "\t" not in line:
            continue
        eng, spa = line.split("\t")
        eng, spa = normalizeString(eng), normalizeString(spa)
        pairs.append((eng, spa))

    seen = set()
    unique_pairs = []
    for eng, spa in pairs:
        if eng not in seen:
            unique_pairs.append((eng, spa))
            seen.add(eng)
    return unique_pairs

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, pairs, reverse=False):
    if reverse:
        pairs = [tuple(reversed(p)) for p in pairs]
        input_lang = LanguageDictionary(lang2)
        output_lang = LanguageDictionary(lang1)
    else:
        input_lang = LanguageDictionary(lang1)
        output_lang = LanguageDictionary(lang2)

    pairs = filterPairs(pairs)
    for eng, spa in pairs:
        input_lang.addSentence(eng)
        output_lang.addSentence(spa)

    return input_lang, output_lang, pairs

class TranslationDataset(Dataset):
    def __init__(self, data_pairs, input_lang, output_lang):
        self.data = data_pairs
        self.input_lang = input_lang
        self.output_lang = output_lang

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eng, spa = self.data[idx]
        eng_idx = [self.input_lang.word2index.get(w, UNK_token) for w in eng.split(' ')]
        spa_idx = [self.output_lang.word2index.get(w, UNK_token) for w in spa.split(' ')]
        spa_idx = [SOS_token] + spa_idx + [EOS_token]
        return torch.tensor(eng_idx), torch.tensor(spa_idx)

def collate_batch(batch):
    eng_batch, spa_input, spa_target = [], [], []
    for eng, spa in batch:
        eng_batch.append(eng)
        spa_input.append(spa[:-1])
        spa_target.append(spa[1:])

    eng_batch = torch.nn.utils.rnn.pad_sequence(eng_batch, batch_first=True, padding_value=PAD_token)
    spa_input = torch.nn.utils.rnn.pad_sequence(spa_input, batch_first=True, padding_value=PAD_token)
    spa_target = torch.nn.utils.rnn.pad_sequence(spa_target, batch_first=True, padding_value=PAD_token)

    eng_lens = torch.tensor([len(seq) for seq in eng_batch])
    spa_lens = torch.tensor([len(seq) for seq in spa_input])
    return eng_batch, spa_input, spa_target, eng_lens, spa_lens

def get_dataloader(batch_size=BATCH_SIZE, reverse=False):
    raw_pairs = load_and_preprocess_data()
    input_lang, output_lang, pairs = prepareData("eng", "spa", raw_pairs, reverse)
    dataset = TranslationDataset(pairs, input_lang, output_lang)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    return dataloader, input_lang, output_lang

if __name__ == "__main__":
    train_dl, input_lang, output_lang = get_dataloader()
    print(f"Vocab sizes: eng={input_lang.n_words}, spa={output_lang.n_words}")
    print(f"Train DataLoader batches: {len(train_dl)}")
