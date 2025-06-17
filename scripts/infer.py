# scripts/infer.py

import torch
from model import GPT, GPTConfig
from transformers import PreTrainedTokenizerFast

# Load tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer/tokenizer.json",
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<s>",
    eos_token="</s>",
)

# Config (must match training)
config = {
    "vocab_size": tokenizer.vocab_size,
    "block_size": 128,
    "n_layer": 2,      # ✅ match training
    "n_head": 2,       # ✅ match training
    "n_embd": 64,      # ✅ match training
}

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(GPTConfig(**config)).to(device)
model.load_state_dict(torch.load("models/gpt_model.pt", map_location=device))
model.eval()

# Prompt
prompt = "In the beginning"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Generate
output = model.generate(input_ids, max_new_tokens=100)[0]
decoded = tokenizer.decode(output.tolist())

print("\n📜 Generated Text:\n")
print(decoded)