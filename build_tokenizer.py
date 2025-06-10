from collections import Counter
import json

# Read corpus
with open('corpus.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()  # Normalize to lowercase
    words = text.split()

# Build vocabulary
vocab_size = 10000
word_counts = Counter(words).most_common(vocab_size - 4)  # Reserve 4 for special tokens
vocab = {word: idx for idx, (word, _) in enumerate(word_counts)}
vocab['<unk>'] = len(vocab)
vocab['<pad>'] = len(vocab)
vocab['<s>'] = len(vocab)
vocab['</s>'] = len(vocab)

# Save vocab
with open('vocab.json', 'w') as f:
    json.dump(vocab, f)
print(f"Vocabulary of size {len(vocab)} saved to vocab.json")