"""
Step 4 – Pre-compute reference log-probs.

Runs π_ref once over every (prompt, chosen) and (prompt, rejected) pair
and saves the results to disk. During preference-optimization training we
load these cached values, so only one model needs to be in memory.

Saves:
  outputs/ref_logprobs/train_ref_logprobs.json
  outputs/ref_logprobs/val_ref_logprobs.json

Run:
    python step4_ref_logprobs.py
"""

import os
import json
import torch
from transformers import GPT2LMHeadModel
from tqdm import tqdm

from utils import (
    SFT_MODEL_DIR, PREF_DATA_DIR, REF_LOGPROBS_DIR,
    MAX_SEQ_LEN, SEED,
    get_tokenizer, tokenize_pair, compute_logprobs,
)

torch.manual_seed(SEED)

BATCH_SIZE = 8


def process_split(model, tokenizer, pairs, out_path):
    results = {}

    for i in tqdm(range(0, len(pairs), BATCH_SIZE), desc=f"  {os.path.basename(out_path)}"):
        batch = pairs[i : i + BATCH_SIZE]

        chosen_tensors   = [tokenize_pair(tokenizer, p["prompt"], p["chosen"])   for p in batch]
        rejected_tensors = [tokenize_pair(tokenizer, p["prompt"], p["rejected"]) for p in batch]

        ch_ids  = torch.stack([t[0] for t in chosen_tensors])
        ch_attn = torch.stack([t[1] for t in chosen_tensors])
        ch_resp = torch.stack([t[2] for t in chosen_tensors])

        rj_ids  = torch.stack([t[0] for t in rejected_tensors])
        rj_attn = torch.stack([t[1] for t in rejected_tensors])
        rj_resp = torch.stack([t[2] for t in rejected_tensors])

        with torch.no_grad():
            ch_logp, ch_len = compute_logprobs(model, ch_ids, ch_attn, ch_resp)
            rj_logp, rj_len = compute_logprobs(model, rj_ids, rj_attn, rj_resp)

        for j, pair in enumerate(batch):
            results[str(pair["idx"])] = {
                "chosen_ref_logp":   ch_logp[j].item(),
                "rejected_ref_logp": rj_logp[j].item(),
                "chosen_len":        ch_len[j].item(),
                "rejected_len":      rj_len[j].item(),
            }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"  Saved {len(results)} entries → {out_path}")


def main():
    print("=" * 60)
    print("Step 4: Pre-computing reference log-probs")
    print("=" * 60)

    tokenizer = get_tokenizer()
    model     = GPT2LMHeadModel.from_pretrained(SFT_MODEL_DIR)
    model.eval()

    for split in ("train", "val"):
        with open(os.path.join(PREF_DATA_DIR, f"{split}.json")) as f:
            pairs = json.load(f)
        out_path = os.path.join(REF_LOGPROBS_DIR, f"{split}_ref_logprobs.json")
        process_split(model, tokenizer, pairs, out_path)


if __name__ == "__main__":
    main()
