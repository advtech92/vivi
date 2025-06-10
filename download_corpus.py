import gutenbergpy.textget
import re
import glob


# Download books by Gutenberg ID
def download_gutenberg_book(book_id, output_file):
    try:
        raw_text = gutenbergpy.textget.get_text_by_id(book_id)
        # Remove headers/footers
        text = re.sub(r'\*\*\*.*?\*\*\*', '', raw_text.decode('utf-8'), flags=re.DOTALL)
        text = re.sub(r'\n+', '\n', text).strip()
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
    except Exception as e:
        print(f"Error downloading book {book_id}: {e}")


# Download selected books
books = [
    (1342, 'pride_and_prejudice.txt'),
    (45, 'anne_of_green_gables.txt'),
    (74, 'tom_sawyer.txt')
]
for book_id, filename in books:
    print(f"Downloading book ID {book_id}...")
    download_gutenberg_book(book_id, filename)

# Combine into corpus
corpus = ''
for file in glob.glob('*.txt'):
    with open(file, 'r', encoding='utf-8') as f:
        corpus += f.read() + '\n'
with open('corpus.txt', 'w', encoding='utf-8') as f:
    f.write(corpus)
print("Corpus created at corpus.txt")