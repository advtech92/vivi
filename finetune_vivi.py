import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import json

# Define model (same as before)
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

# Conversation dataset
class ViviDataset(Dataset):
    def __init__(self, json_file, vocab, max_len=32):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        user = self.data[idx]['user'].lower().split()
        vivi = self.data[idx]['vivi'].lower().split()
        seq = [self.vocab['<s>']] + [self.vocab.get(word, self.vocab['<unk>']) for word in user + vivi] + [self.vocab['</s>']]
        seq = seq[:self.max_len] + [self.vocab['<pad>']] * (self.max_len - len(seq))
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])

# Load vocab and data
with open('vocab.json', 'r') as f:
    vocab = json.load(f)
dataset = ViviDataset('vivi_conversations.json', vocab)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Load model
model = VivianTransformer(len(vocab)).cuda()
model.load_state_dict(torch.load('vivi_base.pt'))
optimizer = optim.Adam(model.parameters(), lr=0.00005)  # Lower LR for fine-tuning
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

# Fine-tune
for epoch in range(10):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.cuda(), tgt.cuda()
        optimizer.zero_grad()
        output = model(src)
        loss = criterion(output.view(-1, len(vocab)), tgt.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Fine-tune Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}')

# Save model
torch.save(model.state_dict(), 'vivi_finetuned.pt')
print("Fine-tuned model saved to vivi_finetuned.pt")