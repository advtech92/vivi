import torch
import torch.nn as nn
import json

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
    print("Warning: CUDA not available. Running on CPU will be slower.")

# Load vocab
try:
    with open('vocab.json', 'r') as f:
        vocab = json.load(f)
except FileNotFoundError:
    print("Error: vocab.json not found. Run build_tokenizer.py first.")
    exit(1)

# Load model
try:
    model = VivianTransformer(len(vocab)).to(device)
    model.load_state_dict(torch.load('vivi_finetuned.pt', map_location=device))
except FileNotFoundError:
    print("Error: vivi_finetuned.pt not found. Trying vivi_base.pt...")
    try:
        model.load_state_dict(torch.load('vivi_base.pt', map_location=device))
    except FileNotFoundError:
        print("Error: vivi_base.pt not found. Run train_vivi.py first.")
        exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)
model.eval()

# Reverse vocab for decoding
id2word = {idx: word for word, idx in vocab.items()}

# Context memory
context_memory = []
memory_size = 5

def generate_response(prompt, max_len=32, p=0.9):
    global context_memory
    context_memory.append(prompt)
    if len(context_memory) > memory_size:
        context_memory = context_memory[-memory_size:]
    input_text = ' '.join(context_memory).lower()
    input_ids = [vocab['<s>']] + [vocab.get(word, vocab['<unk>']) for word in input_text.split()]
    input_tensor = torch.tensor([input_ids], device=device)
    
    with torch.no_grad():
        for _ in range(max_len - len(input_ids)):
            output = model(input_tensor)
            logits = output[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            probs, indices = probs.sort(descending=True)
            cum_probs = torch.cumsum(probs, dim=-1)
            mask = cum_probs <= p
            if not mask.any():
                mask[0] = True
            probs = probs[mask]
            indices = indices[mask]
            next_word_id = torch.multinomial(probs, 1).item()  # Get scalar index
            next_word_tensor = torch.tensor([[indices[next_word_id]]], device=device)
            input_tensor = torch.cat([input_tensor, next_word_tensor], dim=1)
            if indices[next_word_id].item() == vocab['</s>']:
                break

    response_ids = input_tensor[0, len(input_ids):].tolist()
    response = ' '.join(id2word.get(idx, '<unk>') for idx in response_ids if idx != vocab['<pad>'])
    context_memory.append(response)
    return response

# Save conversations
conversations = []
try:
    with open('vivi_conversations.json', 'r') as f:
        conversations = json.load(f)
except FileNotFoundError:
    pass

# Interactive loop
print("Chat with Vivi! Type 'exit' or 'quit' to stop.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    response = generate_response(user_input)
    print(f"Vivi: {response}")
    conversations.append({"user": user_input, "vivi": response})
    with open('vivi_conversations.json', 'w') as f:
        json.dump(conversations, f, indent=2)