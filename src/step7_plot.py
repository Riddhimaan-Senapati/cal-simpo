"""
Step 7 – Generate all figures.

Figures produced:
  fig1_training_dynamics.png  – chosen/rejected reward + margin over training steps
  fig2_performance.png        – grouped bar: oracle reward + perplexity by method
  fig3_reward_accuracy.png    – reward accuracy bar chart
  fig4_training_loss.png      – training loss curves for all methods

Run:
    python step7_plot.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend
import matplotlib.pyplot as plt

from utils import RESULTS_DIR, FIGURES_DIR

METHODS        = ["dpo", "caldpo", "simpo"]
METHOD_LABELS  = {"dpo": "DPO", "caldpo": "Cal-DPO", "simpo": "SimPO", "sft": "SFT"}
COLORS         = {"dpo":    "#1f77b4",
                  "caldpo": "#2ca02c",
                  "simpo":  "#ff7f0e",
                  "sft":    "#9467bd"}


def load_log(method):
    path = os.path.join(RESULTS_DIR, f"{method}_training_log.json")
    if not os.path.isfile(path):
        print(f"  WARNING: no log for {method}")
        return None
    with open(path) as f:
        return json.load(f)


def load_eval():
    path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    if not os.path.isfile(path):
        print("  WARNING: no evaluation_results.json")
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1 – Training dynamics  (instruction §6 Figure 1)
# ---------------------------------------------------------------------------

def fig1_training_dynamics():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax, method in zip(axes, METHODS):
        log = load_log(method)
        label = METHOD_LABELS[method]

        if log is None:
            ax.set_title(label)
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="grey")
            continue

        entries    = [e for e in log if "val_chosen_reward" in e]
        steps      = [e["step"]                for e in entries]
        chosen_r   = [e["val_chosen_reward"]   for e in entries]
        rejected_r = [e["val_rejected_reward"] for e in entries]
        margin     = [e["val_margin"]          for e in entries]

        ax.plot(steps, chosen_r,   "b-",  lw=1.8, label="Chosen reward")
        ax.plot(steps, rejected_r, "r-",  lw=1.8, label="Rejected reward")
        ax.plot(steps, margin,     "g--", lw=1.8, label="Margin")
        ax.axhline(0, color="black", lw=0.8, ls=":", alpha=0.6)

        ax.set_xlabel("Training Steps", fontsize=11)
        ax.set_ylabel("Implicit Reward Value", fontsize=11)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Figure 1: Training Dynamics", fontsize=13, y=1.01)
    plt.tight_layout()
    _save("fig1_training_dynamics.png")


# ---------------------------------------------------------------------------
# Figure 2 – Oracle reward + perplexity comparison  (instruction §6 Figure 2)
# ---------------------------------------------------------------------------

def fig2_performance():
    data = load_eval()
    if data is None:
        return

    order  = ["sft", "dpo", "caldpo", "simpo"]
    data_d = {r["method"]: r for r in data}
    methods = [m for m in order if m in data_d]

    labels = [METHOD_LABELS[m] for m in methods]
    colors = [COLORS[m]         for m in methods]
    x      = np.arange(len(methods))

    rewards = [data_d[m]["oracle_reward_mean"] for m in methods]
    stds    = [data_d[m]["oracle_reward_std"]  for m in methods]
    ppls    = [data_d[m]["perplexity"]         for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x, rewards, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="white", lw=1.2)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel("Oracle Reward Score", fontsize=11)
    ax1.set_title("Oracle Reward  (↑ better)", fontsize=12)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, ppls, color=colors, alpha=0.85, edgecolor="white", lw=1.2)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylabel("Perplexity", fontsize=11)
    ax2.set_title("Perplexity  (↓ better)", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Figure 2: Final Performance Comparison", fontsize=13)
    plt.tight_layout()
    _save("fig2_performance.png")


# ---------------------------------------------------------------------------
# Figure 3 – Reward accuracy  (instruction §6 Figure 3)
# ---------------------------------------------------------------------------

def fig3_reward_accuracy():
    data = load_eval()
    if data is None:
        return

    rows    = [r for r in data if r.get("reward_accuracy") is not None]
    methods = [r["method"]         for r in rows]
    accs    = [r["reward_accuracy"] * 100 for r in rows]
    colors  = [COLORS.get(m, "#555") for m in methods]
    labels  = [METHOD_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(np.arange(len(methods)), accs, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(50, color="grey", ls="--", lw=1, label="Chance (50 %)")
    ax.set_xticks(np.arange(len(methods))); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Reward Accuracy (%)", fontsize=11)
    ax.set_title("Figure 3: Reward Accuracy on Validation Set", fontsize=12)
    ax.set_ylim(40, 100)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save("fig3_reward_accuracy.png")


# ---------------------------------------------------------------------------
# Figure 4 – Training loss curves
# ---------------------------------------------------------------------------

def fig4_training_loss():
    fig, ax = plt.subplots(figsize=(9, 5))
    any_data = False

    for method in METHODS:
        log = load_log(method)
        if log is None:
            continue
        entries = [e for e in log if "train_loss" in e]
        steps   = [e["step"]       for e in entries]
        losses  = [e["train_loss"] for e in entries]
        ax.plot(steps, losses, label=METHOD_LABELS[method],
                color=COLORS[method], lw=1.8)
        any_data = True

    if not any_data:
        print("  No training logs found for fig4.")
        plt.close()
        return

    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel("Training Loss",  fontsize=11)
    ax.set_title("Figure 4: Training Loss Curves", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save("fig4_training_loss.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def main():
    print("=" * 60)
    print("Step 7: Generating figures")
    print("=" * 60)
    fig1_training_dynamics()
    fig2_performance()
    fig3_reward_accuracy()
    fig4_training_loss()
    print(f"\nAll figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
