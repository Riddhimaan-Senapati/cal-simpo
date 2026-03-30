"""
Shared constants, tokenization helpers, and core log-prob computation.
All scripts import from here.
"""

import os
import json
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS = os.path.join(_ROOT, "outputs")

REWARD_MODEL_DIR  = os.path.join(OUTPUTS, "reward_model")
SFT_MODEL_DIR     = os.path.join(OUTPUTS, "sft_model")
PREF_DATA_DIR     = os.path.join(OUTPUTS, "preference_data")
REF_LOGPROBS_DIR  = os.path.join(OUTPUTS, "ref_logprobs")
CHECKPOINTS_DIR   = os.path.join(OUTPUTS, "checkpoints")
RESULTS_DIR       = os.path.join(OUTPUTS, "results")
FIGURES_DIR       = os.path.join(OUTPUTS, "figures")

# ---------------------------------------------------------------------------
# Model / data constants
# ---------------------------------------------------------------------------
BASE_MODEL   = "gpt2"
PROMPT_LEN   = 32    # tokens taken as prompt prefix
MAX_NEW_TOK  = 128   # tokens generated per completion
MAX_SEQ_LEN  = 192   # prompt (32) + response (128) + small buffer; used for training tensors

# ---------------------------------------------------------------------------
# Best hyperparameters (instruction §3 + §4)
# ---------------------------------------------------------------------------
DPO_BETA     = 0.001
CALDPO_BETA  = 0.001
SIMPO_BETA   = 2.0
SIMPO_GAMMA  = 1.0

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------
LR              = 5e-6
MICRO_BATCH     = 8
GRAD_ACCUM      = 4    # effective batch = 32
NUM_EPOCHS      = 5    # more epochs to compensate for small dataset
WARMUP_STEPS    = 30
EVAL_EVERY      = 10   # gradient steps between val-metric logging
ORACLE_EVERY    = 50   # gradient steps between oracle reward evals
SAVE_EVERY      = 50
SEED            = 42

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


# ---------------------------------------------------------------------------
# Core: tokenize a (prompt, response) pair into fixed-length tensors
# ---------------------------------------------------------------------------

def tokenize_pair(tokenizer, prompt: str, response: str, max_len: int = MAX_SEQ_LEN):
    """
    Returns three LongTensors of length max_len:
      input_ids      - token ids (right-padded)
      attention_mask - 1 for real tokens, 0 for padding
      response_mask  - 1 for response tokens only (used to compute log-prob sum)
    """
    prompt_ids   = tokenizer.encode(prompt,   add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)

    # Keep prompt at most PROMPT_LEN tokens; truncate response to fit in max_len
    prompt_ids   = prompt_ids[:PROMPT_LEN]
    max_resp_len = max_len - len(prompt_ids)
    response_ids = response_ids[:max_resp_len]

    full_ids = prompt_ids + response_ids
    pad_len  = max_len - len(full_ids)

    attention_mask = [1] * len(full_ids) + [0] * pad_len
    full_ids       = full_ids            + [tokenizer.pad_token_id] * pad_len
    response_mask  = ([0] * len(prompt_ids)
                      + [1] * len(response_ids)
                      + [0] * pad_len)

    return (
        torch.tensor(full_ids,        dtype=torch.long),
        torch.tensor(attention_mask,  dtype=torch.long),
        torch.tensor(response_mask,   dtype=torch.float),
    )


# ---------------------------------------------------------------------------
# Core: compute sum of log-probs over response tokens
# ---------------------------------------------------------------------------

def compute_logprobs(model, input_ids, attention_mask, response_mask):
    """
    Compute per-example sum of log-probs and response length.

    Gradient flow is determined by the calling context (use torch.no_grad()
    externally when you don't need gradients).

    Args:
        model          : causal LM
        input_ids      : (B, L)
        attention_mask : (B, L)
        response_mask  : (B, L) – float, 1 for response tokens

    Returns:
        logp_sum : (B,) – sum of log-probs over response tokens
        lengths  : (B,) – number of response tokens
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits  = outputs.logits  # (B, L, V)

    # Standard next-token-prediction shift
    shift_logits = logits[:, :-1, :]        # predicts token[i+1]
    shift_labels = input_ids[:, 1:]         # (B, L-1)
    shift_mask   = response_mask[:, 1:]     # 1 where we're predicting a response token

    log_probs       = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    logp_sum = (token_log_probs * shift_mask).sum(-1)
    lengths  = shift_mask.sum(-1)
    return logp_sum, lengths


# ---------------------------------------------------------------------------
# Oracle reward: log p(pos) – log p(neg) from the classifier
# ---------------------------------------------------------------------------

def compute_oracle_reward(reward_model, reward_tokenizer, texts,
                           batch_size: int = 16, device: str = "cpu"):
    """
    Returns a (N,) float tensor of oracle log-odds rewards.
    """
    reward_model.eval()
    rewards = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = reward_tokenizer(
            batch,
            truncation=True, max_length=512,
            padding="max_length", return_tensors="pt"
        )
        with torch.no_grad():
            logits = reward_model(**enc).logits        # (B, 2)
        log_p  = F.log_softmax(logits, dim=-1)
        r      = log_p[:, 1] - log_p[:, 0]            # log p(pos) – log p(neg)
        rewards.append(r.cpu())
    return torch.cat(rewards)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def make_output_dirs():
    for d in [REWARD_MODEL_DIR, SFT_MODEL_DIR, PREF_DATA_DIR,
              REF_LOGPROBS_DIR, CHECKPOINTS_DIR, RESULTS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
