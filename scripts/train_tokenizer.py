from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
from pathlib import Path

def train_tokenizer(
    input_file="data/processed/train.txt",
    vocab_size=30522,
    min_frequency=2,
    save_path="tokenizer"
):
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files=[input_file],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )

    Path(save_path).mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(save_path)

    # ✅ Save as tokenizer.json for compatibility
    tokenizer_json_path = f"{save_path}/tokenizer.json"
    tokenizer.save(tokenizer_json_path)
    print(f"Tokenizer saved to '{save_path}/' including {tokenizer_json_path}")

if __name__ == "__main__":
    train_tokenizer()