# Implementing DPO, Cal-DPO, and SimPO on the IMDb Sentiment Task

## Overview

You are implementing three preference optimization methods — **DPO**, **Cal-DPO**, and **SimPO** — for controlled sentiment generation on the IMDb dataset. The goal is to train GPT-2 Small (124M) to generate positive movie review completions, then compare the methods on oracle reward score, perplexity, and training dynamics. This follows the controlled evaluation setup from Cal-DPO (Xiao et al., 2024, Table 4) and extends it with SimPO (Meng et al., 2024). The original paper uses GPT-2 Large (774M); we use GPT-2 Small for computational feasibility on CPU/limited-GPU hardware. The relative trends between methods should still hold.

All code should be runnable on a CPU-only machine or a single consumer GPU. Training on CPU with GPT-2 Small should take roughly 1–2 hours per method.

---

## Implementation Status

**Code is fully implemented** in `src/`. Run the complete pipeline with:

```bash
pip install -r requirements.txt
cd src
python run_pipeline.py            # full pipeline (all steps, all methods)
python run_pipeline.py --steps 5 --method dpo   # single method only
```

### File layout

```
src/
  utils.py               shared constants, tokenize_pair, compute_logprobs, compute_oracle_reward
  step1_reward_model.py  fine-tune GPT2ForSequenceClassification on IMDb (oracle reward model)
  step2_sft.py           CLM fine-tune GPT-2 on positive reviews (π_ref / SFT baseline)
  step3_pref_data.py     generate 500 preference pairs from SFT model; 90/10 train/val split
  step4_ref_logprobs.py  cache π_ref log-probs for all pairs (run once; only policy in memory during training)
  step5_train.py         shared training loop for DPO / Cal-DPO / SimPO (--method flag)
  step6_evaluate.py      oracle reward, perplexity (frozen GPT-2), reward accuracy
  step7_plot.py          Figures 1–4 saved to outputs/figures/
  run_pipeline.py        master runner; supports --steps and --method flags
outputs/                 created automatically
  reward_model/          saved classifier checkpoint
  sft_model/             saved SFT checkpoint
  preference_data/       train.json, val.json
  ref_logprobs/          cached reference log-probs (JSON)
  checkpoints/<method>/  step_N/ and final/ subdirectories
  results/               <method>_training_log.json, evaluation_results.json
  figures/               fig1–fig4 PNG files
```

### Chosen hyperparameters

| Setting | Value |
|---|---|
| DPO β | 0.001 |
| Cal-DPO β | 0.001 |
| SimPO β / γ | 2.0 / 1.0 |
| Optimizer | RMSprop, lr=5e-6 |
| Effective batch | 32 (micro-batch 8, grad accum 4) |
| Epochs | 5 |
| Linear warmup | 30 steps |
| Sequence length | 192 tokens (32 prompt + 128 response + buffer) |
| Preference pairs | 500 prompts → 450 train / 50 val |
| Oracle eval (during training) | 50 prompts every 50 gradient steps |
| Final eval prompts | 200 (held-out, different seed) |
| Seed | 42 |

### Implementation notes

- **Cal-DPO loss** follows Algorithm 1 in the paper exactly: no β inside the sigmoid, no 0.5 weight on the calibration term. The Cal-DPO repo applies `0.5 * Cal_loss`; we do not.
- **Chosen + rejected** are concatenated in the batch dimension for a single forward pass per gradient step.
- **Reference log-probs** are pre-computed in step 4 and loaded as tensors during training — only the policy model occupies memory during the training loop.
- **Perplexity** is evaluated with the original pretrained `gpt2` checkpoint (never fine-tuned), matching the paper's setup.
- Training logs (loss, val rewards, margin, reward accuracy, oracle reward) are written to JSON after every 10 gradient steps so runs can be inspected or resumed.

---

## Step 1: Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Minimal set (no `trl`, `accelerate`, or `wandb` needed — the training loop is custom):

```
torch>=2.0.0
transformers>=4.40.0
datasets>=2.14.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

Use `gpt2` (the small, 124M parameter version) from HuggingFace as the base model for both the policy and reference models.

---

## Step 2: Data Preparation

### 2a: Train a binary sentiment classifier (the oracle reward model)

1. Load the IMDb dataset from HuggingFace: `datasets.load_dataset("imdb")`
2. Fine-tune a `gpt2` (small) classification head on the binary sentiment labels (positive/negative) using the 25k training split
3. Define the **oracle reward** as the log-odds of the positive class: `r(x, y) = log(p(positive | x, y)) - log(p(negative | x, y))`
4. Save this classifier — it will be used for both constructing preference data and evaluation

> **Implemented in** `src/step1_reward_model.py`. Uses 5,000 training examples (subset for CPU feasibility). Classifier max length: 512 tokens.

### 2b: Train the SFT model

1. Take the IMDb training set and filter to **positive reviews only**
2. Fine-tune `gpt2` (small) on these positive reviews using standard language modeling (next-token prediction)
3. This gives you `π_ref` (the reference/SFT model)
4. Save this checkpoint

> **Implemented in** `src/step2_sft.py`. Uses 5,000 positive reviews, max length 256 tokens, 3 epochs.

### 2c: Construct preference pairs

1. Sample prompts by taking the **first 32 tokens** of reviews from the IMDb test set as prefixes
2. For each prompt `x`, generate **2 completions** from the SFT model using sampling (temperature=1.0, max_new_tokens=128)
3. Score each completion with the oracle reward model
4. The higher-reward completion becomes `y_w` (chosen), the lower becomes `y_l` (rejected)
5. Record the oracle rewards `r(x, y_w)` and `r(x, y_l)` for each pair — Cal-DPO can use these
6. Create a train/validation split (e.g., 90/10)

> **Implemented in** `src/step3_pref_data.py`. Uses 500 prompts (CPU feasible: ~1000 completions to generate). Saves `oracle_reward` fields for both chosen and rejected in each JSON record.

---

## Step 3: Implement the Three Loss Functions

> **Implemented in** `src/step5_train.py` (`loss_dpo`, `loss_caldpo`, `loss_simpo`). Also see `src/step4_ref_logprobs.py` for the pre-computation of reference log-probs.

All three methods share the same training loop structure. The only difference is the loss computation. For each batch of `(x, y_w, y_l)`:

- Compute `log π_θ(y_w | x)` — sum of per-token log probs for the chosen response under the current policy
- Compute `log π_θ(y_l | x)` — same for the rejected response
- Compute `log π_ref(y_w | x)` and `log π_ref(y_l | x)` — same under the frozen reference model (not needed for SimPO)
- Record `|y_w|` and `|y_l|` — the number of tokens in each response

### 3a: DPO

```
chosen_logratios = log π_θ(y_w | x) - log π_ref(y_w | x)
rejected_logratios = log π_θ(y_l | x) - log π_ref(y_l | x)

L_DPO = -log σ(β * (chosen_logratios - rejected_logratios))
```

Hyperparameter: `β ∈ {1e-3, 2e-3, 3e-3, 1e-2, 1e-1}` (paper searches this range; typically β=0.001 works best)

### 3b: Cal-DPO

```
chosen_logratios = log π_θ(y_w | x) - log π_ref(y_w | x)
rejected_logratios = log π_θ(y_l | x) - log π_ref(y_l | x)

# Standard DPO loss (NOTE: no β multiplier inside sigmoid — see Eq. 10)
L_BT = -log σ(chosen_logratios - rejected_logratios)

# Calibration loss — anchors implicit rewards to ±1/(2β)
# With β=0.001, targets are ±500
L_cal = (chosen_logratios - 1/(2β))^2 + (rejected_logratios + 1/(2β))^2

L_CalDPO = L_BT + L_cal
```

Note: Cal-DPO uses the same β for both the contrastive and calibration terms. The DPO objective in Cal-DPO does NOT multiply by β inside the sigmoid (see Eq. 10 in their paper — they absorb it into the calibration target). Be careful to match their exact formulation.

Here is the exact pseudocode from the paper (Algorithm 1):
```python
def loss(chosen_pi_logps, chosen_ref_logps, rejected_pi_logps, rejected_ref_logps, beta):
    chosen_reward = chosen_pi_logps - chosen_ref_logps
    reject_reward = rejected_pi_logps - rejected_ref_logps
    dpo_losses = -F.logsigmoid(chosen_reward - reject_reward)
    cal_losses = F.mse_loss(chosen_reward, 0.5 * 1/beta) + F.mse_loss(reject_reward, -0.5 * 1/beta)
    cal_dpo_losses = dpo_losses + cal_losses
    return cal_dpo_losses
```

Hyperparameter: `β ∈ {1e-3, 2e-3, 3e-3, 1e-2, 1e-1}` (same search range as DPO)

### 3c: SimPO

```
chosen_avg_logp = (1 / |y_w|) * log π_θ(y_w | x)
rejected_avg_logp = (1 / |y_l|) * log π_θ(y_l | x)

L_SimPO = -log σ(β * chosen_avg_logp - β * rejected_avg_logp - γ)
```

No reference model needed. This saves memory.

Hyperparameters: `β ∈ {2.0, 2.5}`, `γ ∈ {0.5, 1.0, 1.5}`

---

## Step 4: Training Loop

> **Implemented in** `src/step5_train.py`. Run with `python step5_train.py --method <dpo|caldpo|simpo>`.

For all methods, use the following shared training configuration (adapted from Cal-DPO Appendix B.1 for GPT-2 Small on CPU):

- **Optimizer**: RMSprop, learning rate `5e-6`
- **Batch size**: 32 (the paper uses 128, but reduce for memory; use gradient accumulation of 4 with micro-batch 8 if needed)
- **Epochs**: 1–3 over the preference dataset
- **Warmup**: Linear warmup from 0 to 5e-6 over 150 steps
- **Sampling temperature**: 1.0 (for data generation)
- **Precision**: FP32 on CPU (FP16/BF16 if you have a GPU)
- **Reference model**: Frozen copy of the SFT model (loaded in eval mode, no gradients). For SimPO, you can skip loading this entirely to save memory.
- **Pre-compute reference log-probs**: Run the reference model once over all preference pairs and save `log π_ref(y_w|x)` and `log π_ref(y_l|x)` to disk. This way you only need one model in memory during training.

**Target numbers to sanity-check against (Cal-DPO Table 4, GPT-2 Large):**
| Method   | Reward ↑ | Perplexity ↓ |
|----------|----------|--------------|
| SFT      | 0.539    | 35.47        |
| PPO      | 0.626    | 35.05        |
| DPO      | 0.617    | 34.21        |
| DPOP     | 0.632    | 35.58        |
| DPO+NLL  | 0.627    | 34.08        |
| Cal-DPO  | 0.645    | 32.31        |

**Note**: These numbers are from GPT-2 Large. Your GPT-2 Small numbers will differ in absolute value (expect lower rewards and higher perplexity), but the **relative ordering** between methods (Cal-DPO > DPO > SFT) should be preserved. If DPO performs worse than SFT, something is wrong.

### Logging (critical for graphs)

At every evaluation step (e.g., every 50 training steps), compute and log the following on a held-out validation batch:

1. **Chosen implicit reward** — the reward signal for y_w under the current policy:
   - DPO / Cal-DPO: `log π_θ(y_w | x) - log π_ref(y_w | x)`
   - SimPO: `(1/|y_w|) * log π_θ(y_w | x)`
2. **Rejected implicit reward** — same as above but for y_l
3. **Margin** — chosen reward minus rejected reward
4. **Training loss** for each component (contrastive loss, calibration loss if applicable)
5. **Oracle reward** — run the trained sentiment classifier on generations from the current policy (sample a small batch of prompts, generate completions, score them). This is expensive, so do it every 200 steps or so.

---

## Step 5: Evaluation

> **Implemented in** `src/step6_evaluate.py`.

After training, evaluate each method on a held-out set of prompts:

### 5a: Oracle Reward Score
1. Take 500 prompts (first 32 tokens of held-out IMDb reviews)
2. Generate one completion per prompt from each trained policy (temperature=0.7, max_new_tokens=128)
3. Score each completion with the oracle reward model
4. Report mean reward ± standard deviation

### 5b: Perplexity
1. Take the same 500 generated completions
2. Compute perplexity using a **separate, pretrained `gpt2`** (small, not fine-tuned) as the evaluation LM
3. Report mean perplexity

### 5c: Reward Accuracy (on validation preference pairs)
1. For each validation pair (x, y_w, y_l), compute the implicit reward of y_w and y_l under the trained policy
2. Reward accuracy = fraction of pairs where the chosen response has higher reward
3. Report this for each method

Run everything over **3 random seeds** and report mean ± std.

> **Current implementation uses 1 seed (SEED=42).** Error bars in Figure 2 use the std across evaluation prompts rather than across seeds.

---

## Step 6: Figures to Produce

> **Implemented in** `src/step7_plot.py`. Figures saved to `outputs/figures/`.

### Figure 1: Training Dynamics (the most important figure)

Create a 1×3 subplot grid, one panel per method (DPO, Cal-DPO, SimPO). In each panel, plot three lines over training steps:

- **Chosen reward** (blue) — the implicit reward of chosen responses on the validation set
- **Rejected reward** (red) — the implicit reward of rejected responses
- **Margin** (green, dashed) — chosen minus rejected

This replicates Cal-DPO's Figure 1 and Figure 3. The key thing to show: in DPO, the chosen reward drops below zero despite the margin increasing. In Cal-DPO, it stays positive. Observe what happens with SimPO.

**Axis labels**: x = "Training Steps", y = "Implicit Reward Value"  
**Title each panel**: "DPO", "Cal-DPO", "SimPO"

### Figure 2: Final Performance Comparison (Bar Chart)

A grouped bar chart comparing the three methods (+ SFT baseline) on two metrics side by side:

- **Oracle reward score** (higher is better)
- **Perplexity** (lower is better)

Include error bars from the 3-seed runs. This replicates Cal-DPO's Table 4 as a visual.

### Figure 3: Reward Accuracy Comparison

A simple bar chart showing reward accuracy (%) on the validation set for each method.

### Figure 4: Training Loss Curves

> **Implemented** — replaces the ablation figures below as Figure 4 in the current code.
> Training loss curves for all three methods over gradient steps.

### Figure 4 (paper): Calibration Target Ablation (for Cal-DPO)

> **Not yet implemented.** Would require running Cal-DPO for each β ∈ {1e-3, 2e-3, 3e-3, 1e-2, 1e-1} and plotting oracle reward vs. β.

A line plot showing oracle reward score (y-axis) vs. β (x-axis) for Cal-DPO. This replicates Cal-DPO's Figure 4 for the IMDb task and helps you understand the sensitivity of the calibration term.

### Figure 5 (paper): SimPO Margin Ablation

> **Not yet implemented.** Would require running SimPO for each γ ∈ {0.5, 1.0, 1.5} and plotting oracle reward vs. γ.

A line plot showing oracle reward score (y-axis) vs. γ (x-axis) for SimPO with β fixed. This replicates SimPO's Figure 3a for your setting.

---

## Step 7: Writeup Structure (4 pages, double-column)

For reference, here is a suggested outline for the final paper:

1. **Introduction** (~0.5 page): Motivate the problem of preference optimization for LLM alignment. State that DPO suffers from reward collapse and reward-generation mismatch. Introduce Cal-DPO and SimPO as complementary fixes. State your research questions.

2. **Background** (~0.75 page): Briefly define RLHF, DPO (Eq. 2 from Rafailov et al.), Cal-DPO (Eq. 10 from Xiao et al.), and SimPO (Eq. 6 from Meng et al.). Keep equations tight.

3. **Experimental Setup** (~0.75 page): Describe the IMDb task, oracle reward model, preference data construction, model size, hyperparameters, and evaluation metrics.

4. **Results** (~1.5 pages): Present Figures 1–5 with analysis. Key narratives to discuss:
   - Does Cal-DPO's calibration prevent chosen reward collapse compared to DPO?
   - Does SimPO's reference-free formulation achieve competitive reward scores?
   - How do the training dynamics differ across methods?
   - Which method gives the best reward-perplexity tradeoff?

5. **Discussion and Conclusion** (~0.5 page): Summarize findings, discuss limitations (single task, GPT-2 Small instead of Large — note that absolute numbers differ but relative trends should generalize), and note that the natural next step is combining calibration with SimPO (Cal-SimPO).

---

## Key References

- Rafailov et al. (2024). Direct Preference Optimization. NeurIPS 2024.
- Xiao et al. (2024). Cal-DPO: Calibrated Direct Preference Optimization. NeurIPS 2024. Code: https://github.com/tengxiao1/Cal-DPO
- Meng et al. (2024). SimPO: Simple Preference Optimization with a Reference-Free Reward. NeurIPS 2024. Code: https://github.com/princeton-nlp/SimPO
- Schulman et al. (2017). Proximal Policy Optimization Algorithms.
- Tajwar et al. (2024). Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data.

---

## Compute Tips for CPU / Limited Hardware

- **Model size**: GPT-2 Small is ~500MB in FP32. Two copies (policy + reference) fit comfortably in 4GB RAM. With pre-computed reference log-probs, you only need one copy during training.
- **Caching reference log-probs**: This is the single most important optimization. Run the reference model once over all training pairs, save the log-probs as a tensor file. Then during training you only load the policy model. This also makes SimPO vs. DPO/Cal-DPO comparisons fairer.
- **Training speed**: Expect ~2-5 seconds per training step on a modern CPU (depends on batch size and sequence length). With ~1000 steps per epoch, that's roughly 30-90 minutes per training run.
- **Reduce evaluation cost**: Generating completions for oracle reward evaluation is the slowest part on CPU. Do intermediate evaluations on only 50 prompts every 200 steps. Full 500-prompt evaluation only at the end.
- **Gradient accumulation**: If batch size 32 causes memory issues, use micro-batch of 8 with 4 accumulation steps.
- **Save frequently**: Save checkpoints and training logs every 200 steps in case of crashes or interruptions.
- **Optional GPU acceleration**: If you get access to any GPU (even a Colab T4), use it for the data generation and reward model steps, which involve many forward passes. The preference optimization training itself is the fastest part.

### Actual runtime profile (CPU, current implementation)

| Step | Bottleneck | Estimated time |
|---|---|---|
| Step 1 (reward model) | 5k examples × 3 epochs | ~30–60 min |
| Step 2 (SFT) | 5k reviews × 3 epochs | ~30–60 min |
| Step 3 (pref data) | 1000 completions at 128 tokens | ~30–60 min |
| Step 4 (ref log-probs) | 2 forward passes × 500 pairs | ~5–10 min |
| Step 5 (training, per method) | ~140 gradient steps | ~15–30 min |
| Step 6 (evaluation) | 200 completions × 4 models | ~30–60 min |
| **Total** | | **~3–5 hours** |

To cut time: reduce `NUM_TRAIN` in steps 1–2, `NUM_PROMPTS` in step 3, or `N_EVAL_PROMPTS` in step 6 — all are constants at the top of each script.