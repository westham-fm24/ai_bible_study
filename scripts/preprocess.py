import os
from pathlib import Path
from tqdm import tqdm

def clean_text(text):
    text = text.replace('\n', ' ').strip()
    return text

def preprocess_raw_texts(input_dir, output_file):
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file in tqdm(sorted(input_dir.glob("*.txt")), desc="Processing"):
            with open(file, 'r', encoding='utf-8') as in_f:
                raw = in_f.read()
                cleaned = clean_text(raw)
                out_f.write(cleaned + '\n')

if __name__ == "__main__":
    preprocess_raw_texts("data/raw", "data/processed/train.txt")