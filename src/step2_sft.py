"""
Step 2 – Train the SFT / reference model.

Fine-tunes GPT-2 Small on positive IMDb reviews only (CLM objective).
The resulting model is π_ref for DPO and Cal-DPO, and the generation
model used to build preference pairs.

Run:
    python step2_sft.py
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm

from utils import BASE_MODEL, SFT_MODEL_DIR, SEED, get_tokenizer

torch.manual_seed(SEED)

# ---- Hyperparams ----
MAX_LEN    = 256
BATCH_SIZE = 8
GRAD_ACCUM = 4
EPOCHS     = 3
LR         = 5e-5
NUM_TRAIN  = 5000   # positive reviews to train on


class CLMDataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        ids = self.input_ids[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def train():
    print("=" * 60)
    print("Step 2: Training SFT (reference) model")
    print("=" * 60)

    dataset   = load_dataset("imdb")
    tokenizer = get_tokenizer()
    tokenizer.padding_side = "right"

    # Positive reviews only
    positives = dataset["train"].filter(lambda x: x["label"] == 1)
    positives = positives.shuffle(seed=SEED).select(range(min(NUM_TRAIN, len(positives))))
    print(f"  Training on {len(positives)} positive reviews")

    enc = tokenizer(
        positives["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt",
    )

    loader = DataLoader(
        CLMDataset(enc["input_ids"]),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    model     = GPT2LMHeadModel.from_pretrained(BASE_MODEL)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_opt_steps = (len(loader) // GRAD_ACCUM) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, 50, total_opt_steps)

    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            out  = model(**batch)
            loss = out.loss / GRAD_ACCUM
            loss.backward()
            running_loss += out.loss.item()

            if (i + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        print(f"  Epoch {epoch+1} | avg loss {running_loss / len(loader):.4f}")

    os.makedirs(SFT_MODEL_DIR, exist_ok=True)
    model.save_pretrained(SFT_MODEL_DIR)
    tokenizer.save_pretrained(SFT_MODEL_DIR)
    print(f"  Saved to {SFT_MODEL_DIR}")


if __name__ == "__main__":
    train()
