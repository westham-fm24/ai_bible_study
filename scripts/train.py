# scripts/train.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from pathlib import Path
import yaml
from model import GPT, GPTConfig
from tqdm import tqdm


# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Load tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer/tokenizer.json",
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<s>",
    eos_token="</s>",
)

vocab_size = tokenizer.vocab_size
block_size = config["block_size"]

# Dataset
class TextDataset(Dataset):
    def __init__(self, file_path):
        text = Path(file_path).read_text(encoding="utf-8")
        tokens = tokenizer.encode(text)  # ✅ FIXED
        self.data = torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.data) - block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+block_size]
        y = self.data[idx+1:idx+1+block_size]
        return x, y

# Prepare dataset and loader
dataset = TextDataset(config["train_data_path"])
loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# Set up model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("🚀 Device:", device)

model_cfg = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=2,    # ⬅️ Lower layers
    n_head=2,     # ⬅️ Fewer heads
    n_embd=64     # ⬅️ Smaller embedding
)


# ✅ Instantiate the model first
model = GPT(model_cfg).to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))

# Training loop
model.train()
for epoch in range(config["num_epochs"]):
    total_loss = 0
    for batch_idx, (x, y) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"📚 Epoch {epoch+1}/{config['num_epochs']} - Loss: {avg_loss:.4f}")

# Save model
Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), config["model_save_path"])
print(f"✅ Model saved to {config['model_save_path']}")