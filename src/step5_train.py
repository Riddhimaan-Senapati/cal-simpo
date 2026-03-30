"""
Step 5 – Preference optimisation training (DPO / Cal-DPO / SimPO).

Shared training loop; only the loss function differs per method.
Logs training metrics to outputs/results/<method>_training_log.json.

Run:
    python step5_train.py --method dpo
    python step5_train.py --method caldpo
    python step5_train.py --method simpo
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2ForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from utils import (
    SFT_MODEL_DIR, REWARD_MODEL_DIR, PREF_DATA_DIR, REF_LOGPROBS_DIR,
    CHECKPOINTS_DIR, RESULTS_DIR,
    MAX_SEQ_LEN, PROMPT_LEN, MAX_NEW_TOK,
    DPO_BETA, CALDPO_BETA, SIMPO_BETA, SIMPO_GAMMA,
    LR, MICRO_BATCH, GRAD_ACCUM, NUM_EPOCHS, WARMUP_STEPS,
    EVAL_EVERY, ORACLE_EVERY, SAVE_EVERY, SEED,
    get_tokenizer, tokenize_pair, compute_logprobs, compute_oracle_reward,
)

torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Loss functions  (instruction §3)
# ---------------------------------------------------------------------------

def loss_dpo(chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp, beta):
    """Standard DPO (Rafailov et al. 2024, Eq. 7)."""
    chosen_ratio   = chosen_logp   - chosen_ref_logp
    rejected_ratio = rejected_logp - rejected_ref_logp
    loss = -F.logsigmoid(beta * (chosen_ratio - rejected_ratio))
    return loss.mean(), chosen_ratio.detach(), rejected_ratio.detach()


def loss_caldpo(chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp, beta):
    """
    Cal-DPO (Xiao et al. 2024, Algorithm 1).

    NOTE: no β inside the sigmoid (absorbed into calibration targets).
    Calibration targets: +1/(2β) for chosen, –1/(2β) for rejected.
    """
    chosen_reward   = chosen_logp   - chosen_ref_logp
    rejected_reward = rejected_logp - rejected_ref_logp

    target_pos = torch.full_like(chosen_reward,   0.5 / beta)
    target_neg = torch.full_like(rejected_reward, -0.5 / beta)

    dpo_loss = -F.logsigmoid(chosen_reward - rejected_reward)
    cal_loss = F.mse_loss(chosen_reward, target_pos) + F.mse_loss(rejected_reward, target_neg)
    total    = (dpo_loss + cal_loss).mean()
    return total, chosen_reward.detach(), rejected_reward.detach()


def loss_simpo(chosen_logp, rejected_logp, chosen_len, rejected_len, beta, gamma):
    """
    SimPO (Meng et al. 2024, Eq. 6). No reference model.
    Rewards are length-normalised average log-probs scaled by β.
    """
    chosen_avg   = chosen_logp   / chosen_len.clamp(min=1)
    rejected_avg = rejected_logp / rejected_len.clamp(min=1)

    logits = beta * chosen_avg - beta * rejected_avg - gamma
    loss   = -F.logsigmoid(logits)

    chosen_reward   = (beta * chosen_avg).detach()
    rejected_reward = (beta * rejected_avg).detach()
    return loss.mean(), chosen_reward, rejected_reward


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PrefDataset(Dataset):
    def __init__(self, pairs, ref_logprobs, tokenizer):
        self.pairs       = pairs
        self.ref         = ref_logprobs   # str(idx) → dict
        self.tokenizer   = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p   = self.pairs[i]
        ref = self.ref[str(p["idx"])]

        ch_ids, ch_attn, ch_resp = tokenize_pair(self.tokenizer, p["prompt"], p["chosen"])
        rj_ids, rj_attn, rj_resp = tokenize_pair(self.tokenizer, p["prompt"], p["rejected"])

        return {
            "chosen_ids":       ch_ids,
            "chosen_attn":      ch_attn,
            "chosen_resp":      ch_resp,
            "rejected_ids":     rj_ids,
            "rejected_attn":    rj_attn,
            "rejected_resp":    rj_resp,
            "chosen_ref_logp":  torch.tensor(ref["chosen_ref_logp"],   dtype=torch.float),
            "rejected_ref_logp":torch.tensor(ref["rejected_ref_logp"], dtype=torch.float),
            "chosen_len":       torch.tensor(ref["chosen_len"],        dtype=torch.float),
            "rejected_len":     torch.tensor(ref["rejected_len"],      dtype=torch.float),
        }


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def val_metrics(policy, loader, method, beta, gamma=None):
    policy.eval()
    chosen_rs, rejected_rs = [], []

    for batch in loader:
        ch_logp, ch_len = compute_logprobs(
            policy, batch["chosen_ids"], batch["chosen_attn"], batch["chosen_resp"])
        rj_logp, rj_len = compute_logprobs(
            policy, batch["rejected_ids"], batch["rejected_attn"], batch["rejected_resp"])

        if method in ("dpo", "caldpo"):
            cr = ch_logp - batch["chosen_ref_logp"]
            rr = rj_logp - batch["rejected_ref_logp"]
        else:  # simpo
            cr = beta * ch_logp / ch_len.clamp(min=1)
            rr = beta * rj_logp / rj_len.clamp(min=1)

        chosen_rs.extend(cr.tolist())
        rejected_rs.extend(rr.tolist())

    policy.train()

    cr_mean = float(np.mean(chosen_rs))
    rr_mean = float(np.mean(rejected_rs))
    acc     = float(np.mean([c > r for c, r in zip(chosen_rs, rejected_rs)]))
    return {
        "val_chosen_reward":   cr_mean,
        "val_rejected_reward": rr_mean,
        "val_margin":          cr_mean - rr_mean,
        "val_reward_acc":      acc,
    }


# ---------------------------------------------------------------------------
# Oracle reward: generate completions and score
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_oracle_eval(policy, tokenizer, reward_model, reward_tokenizer,
                    prompts, n=50):
    policy.eval()
    texts = []
    for prompt in prompts[:n]:
        ids = tokenizer.encode(prompt, return_tensors="pt")
        out = policy.generate(
            ids,
            max_new_tokens=MAX_NEW_TOK,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        texts.append(prompt + completion)

    rewards = compute_oracle_reward(reward_model, reward_tokenizer, texts)
    policy.train()
    return float(rewards.mean())


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(method: str):
    print("=" * 60)
    print(f"Step 5: Training  method={method.upper()}")
    print("=" * 60)

    tokenizer = get_tokenizer()

    # ---------- data ----------
    with open(os.path.join(PREF_DATA_DIR, "train.json")) as f:
        train_pairs = json.load(f)
    with open(os.path.join(PREF_DATA_DIR, "val.json")) as f:
        val_pairs = json.load(f)
    with open(os.path.join(REF_LOGPROBS_DIR, "train_ref_logprobs.json")) as f:
        train_ref = json.load(f)
    with open(os.path.join(REF_LOGPROBS_DIR, "val_ref_logprobs.json")) as f:
        val_ref = json.load(f)

    train_loader = DataLoader(
        PrefDataset(train_pairs, train_ref, tokenizer),
        batch_size=MICRO_BATCH, shuffle=True,
    )
    val_loader = DataLoader(
        PrefDataset(val_pairs, val_ref, tokenizer),
        batch_size=MICRO_BATCH,
    )

    # ---------- policy ----------
    policy = GPT2LMHeadModel.from_pretrained(SFT_MODEL_DIR)
    policy.train()

    # ---------- hyperparams ----------
    beta  = {"dpo": DPO_BETA, "caldpo": CALDPO_BETA, "simpo": SIMPO_BETA}[method]
    gamma = SIMPO_GAMMA  # only used by simpo

    # ---------- optimiser ----------
    optimizer    = torch.optim.RMSprop(policy.parameters(), lr=LR)
    total_steps  = (len(train_loader) // GRAD_ACCUM) * NUM_EPOCHS
    scheduler    = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, max(total_steps, WARMUP_STEPS + 1))

    # ---------- reward model for oracle eval ----------
    reward_model     = GPT2ForSequenceClassification.from_pretrained(REWARD_MODEL_DIR)
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_DIR)
    reward_model.eval()
    val_prompts = [p["prompt"] for p in val_pairs]

    # ---------- logging ----------
    log_path = os.path.join(RESULTS_DIR, f"{method}_training_log.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_entries = []

    def flush_log():
        with open(log_path, "w") as f:
            json.dump(log_entries, f, indent=2)

    global_step = 0
    optimizer.zero_grad()

    for epoch in range(NUM_EPOCHS):
        print(f"\n  Epoch {epoch + 1}/{NUM_EPOCHS}")

        for step_in_epoch, batch in enumerate(tqdm(train_loader, desc="  train")):

            # --- forward pass (concatenate chosen+rejected for one pass) ---
            all_ids  = torch.cat([batch["chosen_ids"],  batch["rejected_ids"]],  dim=0)
            all_attn = torch.cat([batch["chosen_attn"], batch["rejected_attn"]], dim=0)
            all_resp = torch.cat([batch["chosen_resp"], batch["rejected_resp"]], dim=0)

            logp_sum, lengths = compute_logprobs(policy, all_ids, all_attn, all_resp)

            B               = batch["chosen_ids"].shape[0]
            chosen_logp     = logp_sum[:B]
            rejected_logp   = logp_sum[B:]
            chosen_len      = lengths[:B]
            rejected_len    = lengths[B:]

            # --- loss ---
            if method == "dpo":
                loss, ch_r, rj_r = loss_dpo(
                    chosen_logp, rejected_logp,
                    batch["chosen_ref_logp"], batch["rejected_ref_logp"],
                    beta,
                )
            elif method == "caldpo":
                loss, ch_r, rj_r = loss_caldpo(
                    chosen_logp, rejected_logp,
                    batch["chosen_ref_logp"], batch["rejected_ref_logp"],
                    beta,
                )
            else:  # simpo
                loss, ch_r, rj_r = loss_simpo(
                    chosen_logp, rejected_logp,
                    chosen_len, rejected_len,
                    beta, gamma,
                )

            (loss / GRAD_ACCUM).backward()

            # --- gradient step ---
            if (step_in_epoch + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # --- periodic validation metrics ---
                if global_step % EVAL_EVERY == 0:
                    vm = val_metrics(policy, val_loader, method, beta, gamma)
                    entry = {"step": global_step, "train_loss": loss.item(), **vm}

                    # --- oracle reward (expensive) ---
                    if global_step % ORACLE_EVERY == 0:
                        oracle = run_oracle_eval(
                            policy, tokenizer,
                            reward_model, reward_tokenizer,
                            val_prompts, n=50,
                        )
                        entry["oracle_reward"] = oracle
                        print(f"  step {global_step:4d} | loss {loss.item():.4f} | "
                              f"oracle {oracle:.4f} | margin {vm['val_margin']:.4f} | "
                              f"acc {vm['val_reward_acc']:.4f}")

                    log_entries.append(entry)
                    flush_log()

                # --- checkpoint ---
                if global_step % SAVE_EVERY == 0:
                    ckpt = os.path.join(CHECKPOINTS_DIR, method, f"step_{global_step}")
                    os.makedirs(ckpt, exist_ok=True)
                    policy.save_pretrained(ckpt)

    # ---------- final save ----------
    final_dir = os.path.join(CHECKPOINTS_DIR, method, "final")
    os.makedirs(final_dir, exist_ok=True)
    policy.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n  Final model saved → {final_dir}")

    # ---------- final oracle eval on full val set ----------
    print("  Running final oracle evaluation (all val prompts)…")
    final_oracle = run_oracle_eval(
        policy, tokenizer, reward_model, reward_tokenizer,
        val_prompts, n=len(val_prompts),
    )
    log_entries.append({"step": global_step, "final_oracle_reward": final_oracle})
    flush_log()
    print(f"  Final oracle reward: {final_oracle:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["dpo", "caldpo", "simpo"])
    args = parser.parse_args()
    train(args.method)
