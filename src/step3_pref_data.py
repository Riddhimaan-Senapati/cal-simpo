"""
Step 3 – Build preference pairs.

For each of NUM_PROMPTS test-set prompts (first PROMPT_LEN tokens):
  1. Generate 2 completions from the SFT model
  2. Score both with the oracle reward model
  3. Higher reward → chosen (y_w), lower → rejected (y_l)

Saves outputs/preference_data/train.json and val.json.

Run:
    python step3_pref_data.py
"""

import os
import json
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2ForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from utils import (
    SFT_MODEL_DIR, REWARD_MODEL_DIR, PREF_DATA_DIR,
    PROMPT_LEN, MAX_NEW_TOK, SEED,
    get_tokenizer, compute_oracle_reward,
)

torch.manual_seed(SEED)

NUM_PROMPTS  = 500    # prompts → 1000 completions to generate; ~30-60 min on CPU
TRAIN_FRAC   = 0.9
TEMPERATURE  = 1.0    # instruction §2c


def generate_two(sft_model, tokenizer, prompt_ids):
    """Return two decoded completion strings (no prompt prefix)."""
    inp = prompt_ids.unsqueeze(0)  # (1, P)
    with torch.no_grad():
        out = sft_model.generate(
            inp,
            max_new_tokens=MAX_NEW_TOK,
            do_sample=True,
            temperature=TEMPERATURE,
            num_return_sequences=2,
            pad_token_id=tokenizer.eos_token_id,
        )
    # out: (2, P + new_tokens)
    p = prompt_ids.shape[0]
    return [tokenizer.decode(seq[p:], skip_special_tokens=True) for seq in out]


def main():
    print("=" * 60)
    print("Step 3: Building preference pairs")
    print("=" * 60)

    dataset          = load_dataset("imdb")
    tokenizer        = get_tokenizer()
    sft_model        = GPT2LMHeadModel.from_pretrained(SFT_MODEL_DIR)
    sft_model.eval()

    reward_model     = GPT2ForSequenceClassification.from_pretrained(REWARD_MODEL_DIR)
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_DIR)
    reward_model.eval()

    test_data = dataset["test"].shuffle(seed=SEED).select(range(NUM_PROMPTS))

    pairs = []
    for i, example in enumerate(tqdm(test_data, desc="Generating pairs")):
        tokens      = tokenizer.encode(example["text"], add_special_tokens=False)
        prompt_ids  = torch.tensor(tokens[:PROMPT_LEN], dtype=torch.long)
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)

        completions = generate_two(sft_model, tokenizer, prompt_ids)

        # Score full text (prompt + completion) with oracle reward model
        full_texts = [prompt_text + c for c in completions]
        rewards    = compute_oracle_reward(reward_model, reward_tokenizer, full_texts)
        r0, r1     = rewards[0].item(), rewards[1].item()

        if r0 >= r1:
            chosen, rejected = completions[0], completions[1]
            chosen_r, rejected_r = r0, r1
        else:
            chosen, rejected = completions[1], completions[0]
            chosen_r, rejected_r = r1, r0

        pairs.append({
            "idx":             i,
            "prompt":          prompt_text,
            "chosen":          chosen,
            "rejected":        rejected,
            "chosen_reward":   chosen_r,
            "rejected_reward": rejected_r,
        })

        if (i + 1) % 50 == 0:
            chosen_rs  = [p["chosen_reward"]   for p in pairs]
            rejected_rs = [p["rejected_reward"] for p in pairs]
            print(f"  {i+1}/{NUM_PROMPTS} | "
                  f"mean chosen={np.mean(chosen_rs):.3f} "
                  f"mean rejected={np.mean(rejected_rs):.3f} "
                  f"mean margin={np.mean(np.array(chosen_rs)-np.array(rejected_rs)):.3f}")

    # Train / val split
    rng     = np.random.default_rng(SEED)
    indices = rng.permutation(len(pairs)).tolist()
    n_train = int(len(pairs) * TRAIN_FRAC)

    train_pairs = [pairs[j] for j in indices[:n_train]]
    val_pairs   = [pairs[j] for j in indices[n_train:]]

    os.makedirs(PREF_DATA_DIR, exist_ok=True)
    with open(os.path.join(PREF_DATA_DIR, "train.json"), "w") as f:
        json.dump(train_pairs, f)
    with open(os.path.join(PREF_DATA_DIR, "val.json"), "w") as f:
        json.dump(val_pairs, f)

    print(f"\n  Saved {len(train_pairs)} train + {len(val_pairs)} val pairs → {PREF_DATA_DIR}")


if __name__ == "__main__":
    main()
