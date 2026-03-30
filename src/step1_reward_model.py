"""
Step 1 – Train the oracle reward model.

Fine-tunes GPT2ForSequenceClassification on IMDb (binary sentiment).
Oracle reward = log p(positive | x, y) - log p(negative | x, y).

Run:
    python step1_reward_model.py
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2ForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from utils import BASE_MODEL, REWARD_MODEL_DIR, SEED, get_tokenizer

torch.manual_seed(SEED)

# ---- Hyperparams ----
CLASSIFIER_MAX_LEN = 512
BATCH_SIZE         = 16
EPOCHS             = 3
LR                 = 2e-5
NUM_TRAIN          = 5000   # subset for CPU feasibility
NUM_VAL            = 1000


class IMDbDataset(Dataset):
    def __init__(self, encodings, labels):
        self.enc    = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()}, self.labels[idx]


def tokenize(texts, tokenizer):
    return tokenizer(
        texts,
        truncation=True,
        max_length=CLASSIFIER_MAX_LEN,
        padding="max_length",
        return_tensors="pt",
    )


def train():
    print("=" * 60)
    print("Step 1: Training oracle reward model")
    print("=" * 60)

    dataset   = load_dataset("imdb")
    tokenizer = get_tokenizer()
    tokenizer.padding_side = "right"

    train_data = dataset["train"].shuffle(seed=SEED).select(range(NUM_TRAIN))
    val_data   = dataset["test"].shuffle(seed=SEED).select(range(NUM_VAL))

    print(f"  Train: {len(train_data)}  Val: {len(val_data)}")

    train_enc = tokenize(train_data["text"], tokenizer)
    val_enc   = tokenize(val_data["text"],   tokenizer)

    train_loader = DataLoader(
        IMDbDataset(train_enc, torch.tensor(train_data["label"])),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        IMDbDataset(val_enc, torch.tensor(val_data["label"])),
        batch_size=BATCH_SIZE,
    )

    model = GPT2ForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    model.config.pad_token_id = tokenizer.pad_token_id

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = total = 0

        for enc, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            optimizer.zero_grad()
            out  = model(**enc, labels=labels)
            out.loss.backward()
            optimizer.step()

            running_loss += out.loss.item()
            preds    = out.logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total   += len(labels)

        # Validation
        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for enc, labels in val_loader:
                preds  = model(**enc).logits.argmax(-1)
                vc    += (preds == labels).sum().item()
                vt    += len(labels)

        print(f"  Epoch {epoch+1} | train loss {running_loss/len(train_loader):.4f} "
              f"| train acc {correct/total:.4f} | val acc {vc/vt:.4f}")

    os.makedirs(REWARD_MODEL_DIR, exist_ok=True)
    model.save_pretrained(REWARD_MODEL_DIR)
    tokenizer.save_pretrained(REWARD_MODEL_DIR)
    print(f"  Saved to {REWARD_MODEL_DIR}")


if __name__ == "__main__":
    train()
