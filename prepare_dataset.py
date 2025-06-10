import torch
from torch.utils.data import Dataset
import json


class TextDataset(Dataset):
    def __init__(self, corpus_file, vocab, max_len=32):
        self.vocab = vocab
        self.max_len = max_len
        with open(corpus_file, 'r', encoding='utf-8') as f:
            text = f.read().lower().split()
        self.tokens = [self.vocab.get(word, self.vocab['<unk>']) for word in text]

    def __len__(self):
        return len(self.tokens) // self.max_len

    def __getitem__(self, idx):
        start = idx * self.max_len
        seq = self.tokens[start:start + self.max_len]
        if len(seq) < self.max_len:
            seq += [self.vocab['<pad>']] * (self.max_len - len(seq))
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])

# Load vocab
with open('vocab.json', 'r') as f:
    vocab = json.load(f)

# Create dataset
dataset = TextDataset('corpus.txt', vocab)
torch.save(dataset, 'dataset.pt')
print("Dataset saved to dataset.pt")