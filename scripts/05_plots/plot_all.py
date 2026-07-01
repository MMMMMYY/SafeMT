"""
Generate all Stage 2 saliency distribution plots.

Input data:
  - scatter_sdxl_v2.json    (596 valid SDXL inconsistent pairs)
  - scatter_gpt_v2.json     (206 valid GPT inconsistent pairs)
  - scatter_gemini_v2.json  (250 valid Gemini inconsistent pairs)

Output: 16 PDF files in the same directory as this script.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent

sdxl   = json.load(open(HERE / "scatter_sdxl_v2.json"))
gpt    = json.load(open(HERE / "scatter_gpt_v2.json"))
gemini = json.load(open(HERE / "scatter_gemini_v2.json"))

stage_cfg = {
    "level2_mismatch": {
        "label": "Level 2 — Classifier-decided inconsistency (Q16/NudeNet)",
        "color_a": "#4e79a7",
        "color_d": "#e15759",
    },
    "level3_drift": {
        "label": "Level 3 — Semantic drift (CLIP similarity)",
        "color_a": "#4e79a7",
        "color_d": "#e15759",
    },
}


def plot_anchor_shift(ax, values, title, color, show_legend=False, add_sigma=False):
    n = len(values)
    n_rose = sum(1 for x in values if x < 0)
    n_fell = sum(1 for x in values if x > 0)
    mean_v   = np.mean(values)
    median_v = np.median(values)
    mu, sigma = np.mean(values), np.std(values)

    if add_sigma:
        ax.axvspan(mu-2*sigma, mu+2*sigma, alpha=0.10, color="#2ab46e", zorder=0)
        ax.axvspan(mu-sigma,   mu+sigma,   alpha=0.18, color="orange",  zorder=0)

    ax.hist(values, bins=30, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(0,        color="red",    linewidth=1.5, linestyle="--", label="no shift")
    ax.axvline(mean_v,   color="orange", linewidth=1.5, linestyle="-",  label="mean")
    ax.axvline(median_v, color="green",  linewidth=1.5, linestyle=":",  label="median")

    if add_sigma:
        ax.axvline(mu-sigma,   color="darkorange", linewidth=1.1, linestyle=":")
        ax.axvline(mu+sigma,   color="darkorange", linewidth=1.1, linestyle=":")
        ax.axvline(mu-2*sigma, color="#1a9e5a",    linewidth=1.0, linestyle=":")
        ax.axvline(mu+2*sigma, color="#1a9e5a",    linewidth=1.0, linestyle=":")

    ax.set_xlabel("Anchor Shift", fontsize=15)
    ax.set_ylabel("Count", fontsize=15)
    # ax.set_title(title, fontsize=11)
    if show_legend:
        ax.legend(fontsize=12, loc="center right", framealpha=0.5)
    ax.tick_params(axis='both', labelsize=15)

    if add_sigma:
        pct_1s = np.mean((np.array(values) >= mu-sigma) & (np.array(values) <= mu+sigma)) * 100
        pct_2s = np.mean((np.array(values) >= mu-2*sigma) & (np.array(values) <= mu+2*sigma)) * 100
        ax.text(0.03, 0.95,
            f"±1σ: {pct_1s:.0f}%\n±2σ: {pct_2s:.0f}%",
            transform=ax.transAxes, va="top", fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))
    else:
        ax.text(0.03, 0.95,
            f"More salient(<0): {n_rose} ({100*n_rose/n:.1f}%)\n"
            f"No shift (=0):   {n - n_rose - n_fell}\n"
            f"Less salient(>0): {n_fell} ({100*n_fell/n:.1f}%)",
            transform=ax.transAxes, va="top", fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))


def plot_delta(ax, values, title, color, show_legend=False):
    n = len(values)
    n_dom    = sum(1 for x in values if x < 0.5)
    mean_v   = np.mean(values)
    median_v = np.median(values)

    ax.hist(values, bins=30, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(0.5,      color="black",  linewidth=1.5, linestyle="--", label="midpoint")
    ax.axvline(mean_v,   color="orange", linewidth=1.5, linestyle="-",  label="mean")
    ax.axvline(median_v, color="green",  linewidth=1.5, linestyle=":",  label="median")
    ax.set_xlabel("MR Rank", fontsize=15)
    ax.set_ylabel("Count", fontsize=15)
    # ax.set_title(title, fontsize=11)
    if show_legend:
        ax.legend(fontsize=12, loc="center right", framealpha=0.5)
    ax.tick_params(axis='both', labelsize=15)
    ax.text(0.97, 0.95,
        f"Change salient (<0.5): {n_dom} ({100*n_dom/n:.1f}%)\n"
        f"Change less salient (≥0.5): {n-n_dom} ({100*(n-n_dom)/n:.1f}%)",
        transform=ax.transAxes, va="top", ha="right", fontsize=12,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))


def save(fig, name):
    plt.tight_layout()
    path = HERE / name
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── SDXL overall ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.4, 3))
plot_anchor_shift(ax, [d["anchor_shift"] for d in sdxl],
    f"SDXL — Anchor Shift Distribution (n={len(sdxl)})", "#4e79a7")
plt.tight_layout(); save(fig, "sdxl_overall_anchor_shift.pdf")

fig, ax = plt.subplots(figsize=(4.4, 3))
plot_delta(ax, [d["delta_top_rel"] for d in sdxl],
    f"SDXL — Delta-Added Word Dominance Distribution (n={len(sdxl)})", "#e15759")
plt.tight_layout(); save(fig, "sdxl_overall_delta_added.pdf")

# ── SDXL by stage ─────────────────────────────────────────────────────────────
for stage, cfg in stage_cfg.items():
    subset = [d for d in sdxl if d["stage"] == stage]
    n = len(subset)

    fig, ax = plt.subplots(figsize=(4.4, 3))
    plot_anchor_shift(ax, [d["anchor_shift"] for d in subset],
        f"SDXL | {cfg['label']}\nAnchor Shift Distribution (n={n})", cfg["color_a"],
        add_sigma=True)
    plt.tight_layout(); save(fig, f"sdxl_{stage}_anchor_shift.pdf")

    fig, ax = plt.subplots(figsize=(4.4, 3))
    plot_delta(ax, [d["delta_top_rel"] for d in subset],
        f"SDXL | {cfg['label']}\nDelta-Added Word Dominance Distribution (n={n})", cfg["color_d"])
    plt.tight_layout(); save(fig, f"sdxl_{stage}_delta_added.pdf")

# ── GPT by stage ──────────────────────────────────────────────────────────────
for stage, cfg in stage_cfg.items():
    subset = [d for d in gpt if d["stage"] == stage]
    n = len(subset)

    fig, ax = plt.subplots(figsize=(4.4, 3))
    plot_anchor_shift(ax, [d["anchor_shift"] for d in subset],
        f"GPT-Image | {cfg['label']}\nAnchor Shift Distribution (n={n})", cfg["color_a"],
        show_legend=(stage == "level2_mismatch"), add_sigma=True)
    plt.tight_layout(); save(fig, f"gpt_{stage}_anchor_shift.pdf")

    fig, ax = plt.subplots(figsize=(4.4, 3))
    plot_delta(ax, [d["delta_top_rel"] for d in subset],
        f"GPT-Image | {cfg['label']}\nDelta-Added Word Dominance Distribution (n={n})", cfg["color_d"],
        show_legend=(stage == "level2_mismatch"))
    plt.tight_layout(); save(fig, f"gpt_{stage}_delta_added.pdf")

# ── Gemini overall ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.4, 3))
plot_anchor_shift(ax, [d["anchor_shift"] for d in gemini],
    f"Gemini — Anchor Shift Distribution (n={len(gemini)})", "#4e79a7")
plt.tight_layout(); save(fig, "gemini_overall_anchor_shift.pdf")

fig, ax = plt.subplots(figsize=(4.4, 3))
plot_delta(ax, [d["delta_top_rel"] for d in gemini],
    f"Gemini — Delta-Added Word Dominance Distribution (n={len(gemini)})", "#e15759")
plt.tight_layout(); save(fig, "gemini_overall_delta_added.pdf")

# ── Gemini by stage ───────────────────────────────────────────────────────────
for stage, cfg in stage_cfg.items():
    subset = [d for d in gemini if d["stage"] == stage]
    n = len(subset)

    fig, ax = plt.subplots(figsize=(4.4, 3))
    plot_anchor_shift(ax, [d["anchor_shift"] for d in subset],
        f"Gemini | {cfg['label']}\nAnchor Shift Distribution (n={n})", cfg["color_a"], add_sigma=True)
        # show_legend=(stage == "level2_mismatch"), add_sigma=True)
    plt.tight_layout(); save(fig, f"gemini_{stage}_anchor_shift.pdf")

    fig, ax = plt.subplots(figsize=(4.4, 3))
    plot_delta(ax, [d["delta_top_rel"] for d in subset],
        f"Gemini | {cfg['label']}\nDelta-Added Word Dominance Distribution (n={n})", cfg["color_d"])
        # show_legend=(stage == "level2_mismatch"))
    plt.tight_layout(); save(fig, f"gemini_{stage}_delta_added.pdf")

print("\nDone. All 16 plots saved.")
