import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import json

# Define TextDataset
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

# Define model
class VivianTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=2, n_heads=4, d_ff=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        return self.fc_out(x)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Warning: CUDA not available. Training on CPU will be slower.")

# Load vocab
try:
    with open('vocab.json', 'r') as f:
        vocab = json.load(f)
except FileNotFoundError:
    print("Error: vocab.json not found. Run build_tokenizer.py first.")
    exit(1)

# Load dataset
try:
    dataset = torch.load('dataset.pt')
except FileNotFoundError:
    print("Error: dataset.pt not found. Run prepare_dataset.py first.")
    exit(1)
except Exception as e:
    print(f"Error loading dataset.pt: {e}")
    exit(1)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize model
model = VivianTransformer(len(vocab)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

# Train
print("Starting training...")
for epoch in range(5):
    model.train()
    total_loss = 0
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src)
        loss = criterion(output.view(-1, len(vocab)), tgt.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    print(f'Epoch {epoch+1}, Average Loss: {total_loss / len(dataloader):.4f}')

# Save model
torch.save(model.state_dict(), 'vivi_base.pt')
print("Model saved to vivi_base.pt")