"""
Step 6 – Final evaluation of all trained models.

Metrics:
  • Oracle reward  – mean ± std over N_EVAL generated completions
  • Perplexity     – using a frozen pretrained GPT-2 (not fine-tuned)
  • Reward accuracy – fraction of val pairs where chosen implicit reward > rejected

Saves outputs/results/evaluation_results.json and prints a table.

Run:
    python step6_evaluate.py
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2ForSequenceClassification, AutoTokenizer
from tqdm import tqdm

from utils import (
    BASE_MODEL, REWARD_MODEL_DIR, SFT_MODEL_DIR, CHECKPOINTS_DIR,
    PREF_DATA_DIR, REF_LOGPROBS_DIR, RESULTS_DIR,
    MAX_NEW_TOK, PROMPT_LEN, MAX_SEQ_LEN,
    DPO_BETA, CALDPO_BETA, SIMPO_BETA,
    SEED,
    get_tokenizer, tokenize_pair, compute_logprobs, compute_oracle_reward,
)
from datasets import load_dataset

torch.manual_seed(SEED)

METHODS        = ["dpo", "caldpo", "simpo"]
N_EVAL_PROMPTS = 200   # prompts for oracle + perplexity eval
GEN_TEMP       = 0.7
PPL_BATCH      = 8


# ---------------------------------------------------------------------------
# Perplexity with a frozen pretrained GPT-2
# ---------------------------------------------------------------------------

def perplexity(model, tokenizer, texts, max_len=256, batch_size=PPL_BATCH):
    model.eval()
    ppls = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc   = tokenizer(batch, truncation=True, max_length=max_len,
                          padding="max_length", return_tensors="pt")
        ids   = enc["input_ids"]
        attn  = enc["attention_mask"]

        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=attn).logits

        shift_logits = logits[:, :-1, :]
        shift_labels = ids[:, 1:]
        shift_mask   = attn[:, 1:].float()

        log_p     = F.log_softmax(shift_logits, dim=-1)
        tok_lp    = log_p.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        lengths   = shift_mask.sum(-1).clamp(min=1)
        avg_nlp   = -(tok_lp * shift_mask).sum(-1) / lengths
        ppls.extend(torch.exp(avg_nlp).tolist())
    return float(np.mean(ppls))


# ---------------------------------------------------------------------------
# Generate completions from a policy
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_completions(policy, tokenizer, prompts):
    policy.eval()
    full_texts = []
    for prompt in tqdm(prompts, desc="    generating", leave=False):
        ids = tokenizer.encode(prompt, return_tensors="pt")
        out = policy.generate(
            ids,
            max_new_tokens=MAX_NEW_TOK,
            do_sample=True,
            temperature=GEN_TEMP,
            pad_token_id=tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        full_texts.append(prompt + completion)
    return full_texts


# ---------------------------------------------------------------------------
# Reward accuracy on validation preference pairs
# ---------------------------------------------------------------------------

@torch.no_grad()
def reward_accuracy(policy, tokenizer, val_pairs, val_ref, method, beta):
    policy.eval()
    accs = []
    BATCH = 8

    for i in range(0, len(val_pairs), BATCH):
        batch_pairs = val_pairs[i : i + BATCH]

        ch_t   = [tokenize_pair(tokenizer, p["prompt"], p["chosen"])   for p in batch_pairs]
        rj_t   = [tokenize_pair(tokenizer, p["prompt"], p["rejected"]) for p in batch_pairs]

        ch_ids  = torch.stack([t[0] for t in ch_t])
        ch_attn = torch.stack([t[1] for t in ch_t])
        ch_resp = torch.stack([t[2] for t in ch_t])
        rj_ids  = torch.stack([t[0] for t in rj_t])
        rj_attn = torch.stack([t[1] for t in rj_t])
        rj_resp = torch.stack([t[2] for t in rj_t])

        ch_logp, ch_len = compute_logprobs(policy, ch_ids, ch_attn, ch_resp)
        rj_logp, rj_len = compute_logprobs(policy, rj_ids, rj_attn, rj_resp)

        ch_ref = torch.tensor([val_ref[str(p["idx"])]["chosen_ref_logp"]   for p in batch_pairs])
        rj_ref = torch.tensor([val_ref[str(p["idx"])]["rejected_ref_logp"] for p in batch_pairs])
        ch_l   = torch.tensor([val_ref[str(p["idx"])]["chosen_len"]        for p in batch_pairs])
        rj_l   = torch.tensor([val_ref[str(p["idx"])]["rejected_len"]      for p in batch_pairs])

        if method in ("dpo", "caldpo"):
            cr = ch_logp - ch_ref
            rr = rj_logp - rj_ref
        else:  # simpo
            cr = beta * ch_logp / ch_l.clamp(min=1)
            rr = beta * rj_logp / rj_l.clamp(min=1)

        accs.extend((cr > rr).float().tolist())

    return float(np.mean(accs))


# ---------------------------------------------------------------------------
# Evaluate one policy
# ---------------------------------------------------------------------------

def evaluate_policy(name, policy, tokenizer,
                    reward_model, reward_tokenizer,
                    ppl_model, eval_prompts,
                    val_pairs, val_ref,
                    method_key=None):

    print(f"  [{name}] generating completions…")
    texts = generate_completions(policy, tokenizer, eval_prompts)

    print(f"  [{name}] scoring oracle reward…")
    rewards = compute_oracle_reward(reward_model, reward_tokenizer, texts)
    mean_r  = float(rewards.mean())
    std_r   = float(rewards.std())

    print(f"  [{name}] computing perplexity…")
    ppl = perplexity(ppl_model, tokenizer, texts)

    ra = None
    if method_key is not None:
        beta = {"dpo": DPO_BETA, "caldpo": CALDPO_BETA, "simpo": SIMPO_BETA}[method_key]
        print(f"  [{name}] computing reward accuracy…")
        ra = reward_accuracy(policy, tokenizer, val_pairs, val_ref, method_key, beta)

    return {
        "method":              name,
        "oracle_reward_mean":  mean_r,
        "oracle_reward_std":   std_r,
        "perplexity":          ppl,
        "reward_accuracy":     ra,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Step 6: Evaluation")
    print("=" * 60)

    tokenizer = get_tokenizer()

    # Oracle reward model
    reward_model     = GPT2ForSequenceClassification.from_pretrained(REWARD_MODEL_DIR)
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_DIR)
    reward_model.eval()

    # Perplexity model – frozen pretrained GPT-2 (never fine-tuned)
    ppl_model = GPT2LMHeadModel.from_pretrained(BASE_MODEL)
    ppl_model.eval()

    # Held-out evaluation prompts (different shuffle seed from step 3)
    dataset   = load_dataset("imdb")
    rng       = np.random.default_rng(SEED + 1)
    test_data = dataset["test"]
    indices   = rng.permutation(len(test_data))[:N_EVAL_PROMPTS * 2]
    eval_prompts = []
    for idx in indices:
        tokens = tokenizer.encode(test_data[int(idx)]["text"], add_special_tokens=False)
        eval_prompts.append(tokenizer.decode(tokens[:PROMPT_LEN], skip_special_tokens=True))
        if len(eval_prompts) >= N_EVAL_PROMPTS:
            break

    print(f"  Using {len(eval_prompts)} evaluation prompts")

    # Validation preference data
    with open(os.path.join(PREF_DATA_DIR, "val.json")) as f:
        val_pairs = json.load(f)
    with open(os.path.join(REF_LOGPROBS_DIR, "val_ref_logprobs.json")) as f:
        val_ref = json.load(f)

    results = []

    # --- SFT baseline ---
    print("\n  Evaluating SFT baseline…")
    sft = GPT2LMHeadModel.from_pretrained(SFT_MODEL_DIR)
    results.append(evaluate_policy(
        "sft", sft, tokenizer,
        reward_model, reward_tokenizer,
        ppl_model, eval_prompts,
        val_pairs, val_ref,
        method_key=None,
    ))
    del sft

    # --- DPO / Cal-DPO / SimPO ---
    for method in METHODS:
        final_dir = os.path.join(CHECKPOINTS_DIR, method, "final")
        if not os.path.isdir(final_dir):
            print(f"\n  WARNING: no checkpoint at {final_dir}, skipping {method}")
            continue
        print(f"\n  Evaluating {method.upper()}…")
        policy = GPT2LMHeadModel.from_pretrained(final_dir)
        results.append(evaluate_policy(
            method, policy, tokenizer,
            reward_model, reward_tokenizer,
            ppl_model, eval_prompts,
            val_pairs, val_ref,
            method_key=method,
        ))
        del policy

    # --- Save ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # --- Print table ---
    print("\n" + "=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)
    header = f"{'Method':<10}  {'Oracle Reward':>18}  {'Perplexity':>12}  {'Reward Acc':>12}"
    print(header)
    print("-" * 65)
    for r in results:
        ra   = f"{r['reward_accuracy']:.4f}" if r["reward_accuracy"] is not None else "  N/A"
        mean = r["oracle_reward_mean"]
        std  = r["oracle_reward_std"]
        ppl  = r["perplexity"]
        print(f"{r['method']:<10}  {mean:>+8.4f} ± {std:<6.4f}  {ppl:>12.2f}  {ra:>12}")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
